"""
Modal harness for the oxidize-c CUDA target (this dev box has no nvcc / no GPU).

Run from `oxidize-c/`:

    modal run modal_cuda.py --action smoke              # nvidia-smi + nvcc + compute cap
    modal run modal_cuda.py --action build              # make cuda, full nvcc output
    modal run modal_cuda.py --action build --gpu A10G   # any Modal GPU type
    modal run modal_cuda.py --action build --arch sm_80 # override NVCCFLAGS -arch
    modal run modal_cuda.py --action test               # CPU `make test` + cuda binary smoke
    modal run modal_cuda.py --action gputest            # CPU-vs-CUDA logit equivalence ON THE GPU
    modal run modal_cuda.py --action bench --model /models/x.gguf

Only the oxidize-c/ source tree is uploaded, plus the one fixture the CPU test
suite needs (`../oxidize-core/tests/fixtures/valid-v3.gguf`), mounted so the
Makefile's relative path still resolves. `build/` lives in a Volume, so repeat
runs are incremental.

Default GPU is T4 (cheapest); the point is proving multi-arch, not speed.
"""

import modal

ROOT = "/workspace/oxidize-c"
FIXTURE = "/workspace/oxidize-core/tests/fixtures/valid-v3.gguf"

IGNORE = [
    "build/**",
    "build-*/**",
    "**/*.o",
    "**/*.a",
    "**/*.so",
    "**/*.gguf",
    "**/*.log",
    "**/*.txt",
    "oxidize-c",  # the built binary (dir of same name is src/, not this)
    "oxidize-c-cuda",
    "oxidize-c-merge",
    "oxidize-c-prune",
    "oxidize-c-quantize",
    "oxidize-c-requant",
    "models/**",
    ".git/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("build-essential", "make", "gcc")
    .add_local_dir(".", ROOT, ignore=IGNORE, copy=False)
    .add_local_file("../oxidize-core/tests/fixtures/valid-v3.gguf", FIXTURE, copy=False)
)

build_cache = modal.Volume.from_name("oxidize-c-cuda-build", create_if_missing=True)

app = modal.App("oxidize-c-cuda")

COMMON = dict(
    image=image,
    gpu="T4",
    volumes={f"{ROOT}/build": build_cache},
    cpu=8.0,
    memory=16384,
    timeout=1800,
)


def _run(cmd: str) -> int:
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in ("CFLAGS", "CXXFLAGS")}
    print(f"\n\033[1;36m$ {cmd}\033[0m", flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=env).returncode


def _make(target: str, arch: str) -> int:
    """`make <target>`, forcing a recompile when -arch changes.

    The Makefile keys build/gemma4_cuda.o off sources only, so a cached object
    from a previous -arch would silently relink into a "successful" build for
    the wrong SM. Nuke it whenever an arch is requested.

    No --arch => the multi-arch fatbin the Makefile ships (sm_70..sm_90 + PTX),
    which is the thing that must run on a T4 as well as an H100.
    """
    # The source tree is a read-only Modal mount and build/ is a Volume: the
    # cached objects come back with LATER mtimes than the freshly mounted
    # sources, so make would happily link a stale gemma4_cuda.o against new code
    # (it did, once — that is how this comment got here). Touch the sources.
    _run("touch src/*.c src/*.h src/cuda/* tests/*.c tests/*.h")
    if arch:
        _run("rm -f build/gemma4_cuda.o build/oxidize-c-cuda-test oxidize-c-cuda")
        return _run(f'make {target} -j8 NVCCFLAGS="-O3 -arch={arch}" 2>&1')
    return _run(f"make {target} -j8 2>&1")


def _make_cuda(arch: str) -> int:
    return _make("cuda", arch)


@app.function(**COMMON)
def smoke() -> str:
    _run("nvidia-smi")
    _run("nvcc --version")
    _run(
        "nvidia-smi --query-gpu=name,compute_cap,memory.total "
        "--format=csv,noheader"
    )
    return "smoke ok"


@app.function(**COMMON)
def build(arch: str = "") -> str:
    _run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    rc = _make_cuda(arch)
    build_cache.commit()
    if rc != 0:
        return f"BUILD FAILED (rc={rc}) — nvcc output above is the real error"
    _run("ls -l oxidize-c-cuda")
    return "build ok"


@app.function(**COMMON)
def test(arch: str = "") -> str:
    rc_cpu = _run("make test -j8 2>&1")
    rc_cuda = _make_cuda(arch)
    if rc_cuda == 0:
        _run("./oxidize-c-cuda --help 2>&1 | head -30")
    build_cache.commit()
    return f"cpu test rc={rc_cpu}, cuda build rc={rc_cuda}"


@app.function(**COMMON)
def gputest(arch: str = "") -> str:
    """The acceptance gate: CPU forward vs CUDA forward, same model, real GPU."""
    _run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    if _make("build/oxidize-c-cuda-test", arch) != 0:
        build_cache.commit()
        return "BUILD FAILED — nvcc output above"
    rc = _run("./build/oxidize-c-cuda-test 2>&1 | grep -v '^gemma4:'")
    build_cache.commit()
    return f"gputest rc={rc} ({'PASS' if rc == 0 else 'FAIL'})"


@app.function(**COMMON)
def bench(model: str = "", arch: str = "") -> str:
    import os

    if not model or not os.path.exists(model):
        return (
            f"no model at {model!r} on the Modal box — nothing to bench. "
            "Upload a GGUF to a Volume and pass --model /path/in/volume."
        )
    if _make_cuda(arch) != 0:
        return "cuda build failed; cannot bench"
    rc = _run(f'./oxidize-c-cuda --model {model} --prompt "hello" --max-tokens 32 --bench')
    return f"bench rc={rc}"


@app.local_entrypoint()
def main(action: str = "smoke", gpu: str = "T4", arch: str = "", model: str = ""):
    fns = {"smoke": smoke, "build": build, "test": test, "gputest": gputest,
           "bench": bench}
    fn = fns[action].with_options(gpu=gpu)
    if action == "smoke":
        print(fn.remote())
    elif action == "bench":
        print(fn.remote(model, arch))
    else:
        print(fn.remote(arch))
