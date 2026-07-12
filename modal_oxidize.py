"""Modal GPU deployment for oxidize-c (Qwythos-9B, qwen3.5 hybrid).

Deploy:  modal deploy modal_oxidize.py
Serves the oxidize-c HTTP/WebSocket server on an L40S with FP16 weights
resident on the GPU (cuBLAS gemv offload). Model GGUF is cached in a Volume.
"""

import subprocess

import modal

MODEL_URL = (
    "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/"
    "resolve/main/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
)
MODEL_PATH = "/models/Qwythos-9B-MTP-Q4_K_M.gguf"

app = modal.App("oxidize-c-qwythos")
vol = modal.Volume.from_name("qwythos-gguf", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "wget")
    .add_local_dir("oxidize-c", "/src", copy=True)
    .run_commands("cd /src && make cuda CC=gcc && ls -la oxidize-c-cuda")
)


def ensure_model() -> None:
    import os
    import uuid

    if os.path.exists(MODEL_PATH):
        return
    partial_path = f"{MODEL_PATH}.{uuid.uuid4().hex}.part"
    try:
        subprocess.run(["wget", "-q", "-O", partial_path, MODEL_URL], check=True)
        os.replace(partial_path, MODEL_PATH)
        vol.commit()
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": vol},
    timeout=3600,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(8090, startup_timeout=900)
def serve():
    import os

    ensure_model()
    if not os.getenv("OXIDIZE_API_KEY") and not os.getenv("OXIDIZE_API_KEYS"):
        raise RuntimeError("missing OXIDIZE_API_KEY or OXIDIZE_API_KEYS")
    subprocess.Popen(
        [
            "/src/oxidize-c-cuda",
            "--model", MODEL_PATH,
            "--serve", "--host", "0.0.0.0", "--port", "8090",
            "--temperature", "0.7", "--ctx", "16384",
        ]
    )


@app.function(image=image, gpu="A10G", volumes={"/models": vol}, timeout=1800)
def debug():
    """One-shot: load model on GPU + generate a few tokens, stderr visible."""
    import subprocess
    ensure_model()
    r = subprocess.run(
        ["/src/oxidize-c-cuda", "--model", MODEL_PATH, "--prompt",
         "The capital of France is a beautiful city with a long history and", "--max-tokens", "64", "--ctx", "8192"],
        capture_output=True, text=True, timeout=1500,
    )
    print("=== STDOUT ===\n", r.stdout)
    print("=== STDERR ===\n", r.stderr)
    print("=== RC ===", r.returncode)


@app.function(image=image, gpu="A10G", volumes={"/models": vol}, timeout=1800)
def debug_serve():
    """Start --serve, probe with urllib (curl absent), capture real behavior."""
    import subprocess, time, json, urllib.request, socket
    ensure_model()
    p = subprocess.Popen(
        ["/src/oxidize-c-cuda", "--model", MODEL_PATH, "--serve",
         "--host", "127.0.0.1", "--port", "8090", "--ctx", "8192"],
        stderr=subprocess.STDOUT)
    # wait for port to open
    up = False
    for _ in range(200):
        if p.poll() is not None:
            print("SERVER EXITED early rc=", p.returncode); return
        try:
            s_ = socket.create_connection(("127.0.0.1", 8090), timeout=1); s_.close()
            up = True; break
        except OSError:
            time.sleep(1)
    print("port open:", up)
    body = json.dumps({"messages":[{"role":"user","content":"Capital of Japan? One word."}],
                       "max_tokens":16,"temperature":0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    try:
        t=time.time()
        resp = urllib.request.urlopen(req, timeout=120).read().decode()
        print(f"OK in {time.time()-t:.1f}s:", resp[:500])
    except Exception as e:
        print("REQUEST FAILED:", repr(e))
    print("server alive:", p.poll() is None)
    p.terminate()
