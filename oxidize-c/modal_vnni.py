"""Modal CPU harness for the AVX-512-VNNI int8-activation kernels.

The dev box is AVX2-only, so the VNNI path (src/quant_vnni.c) cannot execute
there and `make test`'s q8 test self-skips. Run it on a Modal CPU instead:

    modal run modal_vnni.py                 # cpuinfo + make test + tests/vnni_diag.c

Prints /proc/cpuinfo flags first: if avx512vnni is absent the run says so and
the rest is meaningless.
"""

import modal

ROOT = "/workspace/oxidize-c"
FIXTURE = "/workspace/oxidize-core/tests/fixtures/valid-v3.gguf"

IGNORE = [
    "build/**", "build-*/**", "**/*.o", "**/*.a", "**/*.so", "**/*.gguf",
    "**/*.log", "**/*.txt", "oxidize-c", "oxidize-c-cuda", "oxidize-c-merge",
    "oxidize-c-prune", "oxidize-c-quantize", "oxidize-c-requant", "models/**",
    ".git/**",
]

image = (
    modal.Image.debian_slim()
    .apt_install("build-essential", "make", "gcc")
    .add_local_dir(".", ROOT, ignore=IGNORE, copy=False)
    .add_local_file("../oxidize-core/tests/fixtures/valid-v3.gguf", FIXTURE, copy=False)
)

app = modal.App("oxidize-c-vnni")


def _run(cmd: str) -> int:
    import subprocess
    print(f"\n\033[1;36m$ {cmd}\033[0m", flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT).returncode


@app.function(image=image, cpu=8.0, memory=8192, timeout=1800)
def flags() -> str:
    import subprocess
    out = subprocess.run("grep -m1 ^flags /proc/cpuinfo", shell=True,
                         capture_output=True, text=True).stdout
    return ("VNNI " if "avx512vnni" in out else "no-vnni ") + " ".join(
        f for f in out.split() if f.startswith("avx512") or f == "avx2")


@app.function(image=image, cpu=8.0, memory=8192, timeout=1800)
def vnni() -> str:
    _run("grep -m1 'model name' /proc/cpuinfo")
    _run("grep -m1 ^flags /proc/cpuinfo | tr ' ' '\\n' | grep -E "
         "'^(avx2|avx512f|avx512bw|avx512vl|avx512dq|avx512_vnni)$' | sort | tr '\\n' ' '; echo")
    has = _run("grep -qm1 avx512_vnni /proc/cpuinfo") == 0
    if not has:
        return "NO avx512vnni on this Modal instance -- rerun / different region"
    rc = _run("make test -j8 2>&1")
    return f"make test rc={rc}"


@app.local_entrypoint()
def main(probe: int = 0):
    if probe:
        # Modal does not let you pick a CPU model: probe until a VNNI host lands.
        for i in range(probe):
            print(i, flags.remote())
        return
    print(vnni.remote())
