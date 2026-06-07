from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from oxidize_python import cli
from oxidize_python.internal.gguf.types import MetadataType, MetadataValue, TensorInfo
from oxidize_python.internal.gguf.writer import WriterHeader, encode


def _write_minimal_gguf(path: Path) -> None:
    header = WriterHeader(
        version=3,
        metadata={
            "general.architecture": MetadataValue(type=MetadataType.STRING, string="demo"),
        },
        tensors=[TensorInfo(name="weight", dimensions=[1], ggml_type=0, relative_offset=0)],
        alignment=32,
    )
    path.write_bytes(encode(header, bytes([1, 2, 3, 4])))


def test_help_lists_chat() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["help"]) == 0
    assert "chat" in buf.getvalue()


def test_run_help_lists_backend_flags() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["run", "--help"]) == 0
    out = buf.getvalue()
    assert "--backend" in out
    assert "--draft-model" in out
    assert "--ctx-size" in out


def test_chat_help_lists_temperature() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["chat", "--help"]) == 0
    assert "--temperature" in buf.getvalue()
    assert "--top-k" in buf.getvalue()


def test_inspect_minimal_gguf(tmp_path: Path) -> None:
    path = tmp_path / "encoded.gguf"
    _write_minimal_gguf(path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["inspect", str(path)]) == 0
    out = buf.getvalue()
    assert "Tensors in" in out
    assert "general.architecture" in out


def test_list_qwen_models_dir() -> None:
    from oxidize_python.testutil import QWEN_MODEL_ID, qwen_model_path

    models_dir = qwen_model_path().parent
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["list", "--models-dir", str(models_dir)]) == 0
    assert QWEN_MODEL_ID in buf.getvalue()


def test_run_qwen_prompt() -> None:
    from oxidize_python.testutil import (
        assert_generation_text,
        qwen_model_path,
        require_slow_tests,
    )

    require_slow_tests()
    path = qwen_model_path()
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(
            [
                "run",
                str(path),
                "--prompt",
                "Write a Python function that returns the factorial of n.",
                "--max-tokens",
                "32",
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
            ]
        )
    assert code == 0
    assert_generation_text(buf.getvalue())


def test_run_unknown_backend_exits_nonzero() -> None:
    assert cli.main(["run", "model.gguf", "hi", "--backend", "quantum"]) == 1


def test_list_empty_models_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["list"]) == 0
    assert "NAME" in buf.getvalue()


def test_server_model_not_found(tmp_path: Path) -> None:
    from http.client import HTTPConnection
    from threading import Thread

    from oxidize_python.internal.server import listen

    _write_minimal_gguf(tmp_path / "known.gguf")
    port = 18765
    thread = Thread(
        target=listen,
        kwargs={"host": "127.0.0.1", "port": port, "models_dir": str(tmp_path)},
        daemon=True,
    )
    thread.start()
    import time

    time.sleep(0.2)
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    body = json.dumps({"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]})
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 404
    payload = json.loads(resp.read().decode())
    assert "error" in payload


def test_schema_max_tokens_or() -> None:
    from oxidize_python.internal.api.schema import ChatCompletionRequest

    req = ChatCompletionRequest(max_tokens=50)
    assert req.max_tokens_or(128) == 50
    req2 = ChatCompletionRequest(max_completion_tokens=32)
    assert req2.max_tokens_or(128) == 32


def test_loader_config_fields() -> None:
    from oxidize_python.cli_flags import RunOptions

    opts = RunOptions(backend="cuda", n_gpu_layers=10, gpus="0", threads=4, ctx_size=4096)
    cfg = opts.loader_config()
    assert cfg.backend == "cuda"
    assert cfg.n_gpu_layers == 10
    assert cfg.threads == 4
