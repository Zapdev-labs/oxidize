from pathlib import Path


def test_splitk_actions_honor_requested_gpu() -> None:
    source = Path("modal_app.py").read_text()

    assert "gpu_splitk_bench.with_options(gpu=gpu)" in source
    assert "gpu_splitk_test.with_options(gpu=gpu)" in source
