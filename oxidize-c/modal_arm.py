"""Modal harness that proves the NEON kernels on real aarch64 machine code.

The dev box is x86-64 AVX2-only, so src/quant_neon.c cannot execute there. This
app builds the whole tree for aarch64 and runs `make test` — the scalar-vs-NEON
differential in tests/test_quant.c — on genuine ARM instructions.

    modal run modal_arm.py                 # cross-build + make test under qemu
    modal run modal_arm.py --action smoke   # toolchain + a tiny NEON program

WHY qemu and not Graviton: pinning an ARM instance type on this Modal account is
refused server-side ("Specifying _instance_types is not allowed"), and the image
builder only targets amd64 (an arm64 base image fails to pull). So we cross-
compile with aarch64-linux-gnu-gcc and run the aarch64 binaries under
qemu-user-static. qemu emulates ASIMD/NEON (and reports it via getauxval), so the
differential is a real correctness proof of the kernels on aarch64 — what it does
NOT prove is Graviton wall-clock speed. `uname -m` under qemu returns aarch64.

Only the oxidize-c/ tree is uploaded, plus the one CPU-test fixture, mounted so
the Makefile's relative path (../oxidize-core/...) still resolves.
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
    .apt_install("make", "file", "gcc-aarch64-linux-gnu", "g++-aarch64-linux-gnu",
                 "qemu-user-static")
    .add_local_dir(".", ROOT, ignore=IGNORE, copy=False)
    .add_local_file("../oxidize-core/tests/fixtures/valid-v3.gguf", FIXTURE, copy=False)
)

app = modal.App("oxidize-c-arm")

CROSS = "aarch64-linux-gnu-gcc"
# qemu-user needs the aarch64 loader + libs; the cross package installs them here.
QEMU_ENV = {"QEMU_LD_PREFIX": "/usr/aarch64-linux-gnu"}
BUILD = "build-arm64"
MAKE = f"make BUILD={BUILD} CC={CROSS} RUNNER=qemu-aarch64-static"


def _run(cmd: str) -> int:
    import os
    import subprocess
    print(f"\n\033[1;36m$ {cmd}\033[0m", flush=True)
    env = {**os.environ, **QEMU_ENV}
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=env).returncode


@app.function(image=image, cpu=8.0, memory=8192, timeout=1800)
def smoke() -> str:
    _run("uname -m")  # the emulation host: x86_64
    _run(f"{CROSS} -dumpmachine")
    # A tiny NEON program, cross-compiled and run under qemu: uname -> aarch64.
    prog = (
        '#include <arm_neon.h>\n#include <sys/utsname.h>\n#include <stdio.h>\n'
        'int main(){struct utsname u; uname(&u);'
        'float32x4_t a=vdupq_n_f32(2.0f), b=vdupq_n_f32(2.5f);'
        'printf("uname -m = %s ; neon 4*(2*2.5) = %g\\n", u.machine,'
        ' vaddvq_f32(vmulq_f32(a,b))); return 0;}'
    )
    with open(f"{ROOT}/_neon_smoke.c", "w") as f:
        f.write(prog)
    rc = _run(f"{CROSS} -O2 _neon_smoke.c -o _neon_smoke && "
              "qemu-aarch64-static ./_neon_smoke")
    _run("rm -f _neon_smoke _neon_smoke.c")
    return f"smoke rc={rc}"


@app.function(image=image, cpu=8.0, memory=8192, timeout=1800)
def neon() -> str:
    _run("uname -m")
    _run(f"{CROSS} -dumpmachine")
    if _run(f"{MAKE}") != 0:
        return "CROSS BUILD FAILED — output above"
    _run(f"file {BUILD}/oxidize-c-test | sed 's#{ROOT}/##'")
    # The differential: scalar vs NEON over every type, on aarch64 instructions.
    # OC_THREADS=1 keeps the spin-barrier sane under emulation. The suite prints
    # the bound ISA itself (active=neon+dotprod / isa=neon+dotprod).
    rc = _run(f'{MAKE} TEST_ENV="OC_THREADS=1" test')
    return f"make test (aarch64/NEON under qemu) rc={rc} " \
           f"({'PASS' if rc == 0 else 'FAIL'})"


@app.local_entrypoint()
def main(action: str = "neon"):
    print({"smoke": smoke, "neon": neon}[action].remote())
