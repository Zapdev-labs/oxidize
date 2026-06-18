"""Shared CLI flags mirroring oxidize-golang/internal/cli/flags.go."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from oxidize_python.core.backend import parse_backend
from oxidize_python.core.model.loader import LoaderConfig
from oxidize_python.internal.generate.runtime import RunConfig


@dataclass
class RunOptions:
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    threads: int = 0
    ctx_size: int = 0
    backend: str = "cpu"
    n_gpu_layers: int = 0
    gpus: str = ""
    draft_model: str = ""
    draft_tokens: int = 4
    hf_file: str = ""
    use_paged: bool = False
    dflash_fusion: bool = False
    layer_wise: bool = False
    layer_cache: int = 1
    ram_offload: bool = False
    mesh: bool = False
    mesh_port: int = 0
    pipe_head: bool = False
    pipe_tail: bool = False
    pipe_peer: str = ""
    pipe_listen: str = ""
    profile: bool = False
    vision: bool = False
    image: str = ""
    auto_tune: bool = True
    print_plan: str = "auto"

    def loader_config(self) -> LoaderConfig:
        cfg = LoaderConfig()
        cfg.backend = self.backend
        cfg.n_gpu_layers = self.n_gpu_layers
        cfg.gpus = self.gpus
        cfg.threads = self.threads
        cfg.ctx_size = self.ctx_size
        cfg.draft_model = self.draft_model
        cfg.draft_tokens = self.draft_tokens
        return cfg

    def run_config(self, model_path: str) -> RunConfig:
        return RunConfig(
            model_path=model_path,
            prompt=self.prompt.strip(),
            max_new_tokens=self.max_tokens if self.max_tokens > 0 else 128,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            draft_model_path=self.draft_model.strip(),
            draft_tokens_per_step=self.draft_tokens,
            loader=self.loader_config(),
            use_paged=self.use_paged,
            use_dflash_fusion=self.dflash_fusion,
            layer_wise=self.layer_wise,
            layer_cache=self.layer_cache if self.layer_cache > 0 else 4,
            vision=self.vision,
            image_path=self.image.strip(),
        )


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=0)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--draft-model", default="")
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--file", default="", dest="hf_file")
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--dflash-fusion", action="store_true")
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--mesh-port", type=int, default=0)
    parser.add_argument("--pipe-head", action="store_true")
    parser.add_argument("--pipe-tail", action="store_true")
    parser.add_argument("--pipe-peer", default="")
    parser.add_argument("--pipe-listen", default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--image", default="")
    parser.add_argument("--auto", dest="auto_tune", action="store_true")
    parser.add_argument("--no-auto", dest="auto_tune", action="store_false")
    parser.set_defaults(auto_tune=True)
    parser.add_argument("--print-plan", default="auto")
    parser.add_argument("--layer-wise", action="store_true")
    parser.add_argument("--layer-cache", type=int, default=1)
    parser.add_argument("--ram-offload", action="store_true")


def options_from_namespace(
    ns: argparse.Namespace, positional: list[str]
) -> tuple[RunOptions, list[str]]:
    prompt = str(getattr(ns, "prompt", "") or "").strip()
    if not prompt and len(positional) > 1:
        prompt = " ".join(positional[1:])
        positional = positional[:1]
    backend = str(getattr(ns, "backend", "cpu") or "cpu")
    try:
        parse_backend(backend)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    return (
        RunOptions(
            prompt=prompt,
            max_tokens=int(getattr(ns, "max_tokens", 128)),
            temperature=float(getattr(ns, "temperature", 0.8)),
            top_p=float(getattr(ns, "top_p", 0.9)),
            top_k=int(getattr(ns, "top_k", 0)),
            threads=int(getattr(ns, "threads", 0)),
            ctx_size=int(getattr(ns, "ctx_size", 0)),
            backend=backend,
            n_gpu_layers=int(getattr(ns, "n_gpu_layers", 0)),
            gpus=str(getattr(ns, "gpus", "") or ""),
            draft_model=str(getattr(ns, "draft_model", "") or ""),
            draft_tokens=int(getattr(ns, "draft_tokens", 4)),
            hf_file=str(getattr(ns, "hf_file", "") or ""),
            use_paged=bool(getattr(ns, "paged", False)),
            dflash_fusion=bool(getattr(ns, "dflash_fusion", False)),
            mesh=bool(getattr(ns, "mesh", False)),
            mesh_port=int(getattr(ns, "mesh_port", 0)),
            pipe_head=bool(getattr(ns, "pipe_head", False)),
            pipe_tail=bool(getattr(ns, "pipe_tail", False)),
            pipe_peer=str(getattr(ns, "pipe_peer", "") or ""),
            pipe_listen=str(getattr(ns, "pipe_listen", "") or ""),
            profile=bool(getattr(ns, "profile", False)),
            vision=bool(getattr(ns, "vision", False)),
            image=str(getattr(ns, "image", "") or ""),
            auto_tune=bool(getattr(ns, "auto_tune", True)),
            print_plan=str(getattr(ns, "print_plan", "auto") or "auto"),
            layer_wise=bool(getattr(ns, "layer_wise", False)),
            layer_cache=int(getattr(ns, "layer_cache", 1)),
            ram_offload=bool(getattr(ns, "ram_offload", False)),
        ),
        positional,
    )


COMMON_RUN_HELP = (
    "Common options: --prompt, --max-tokens, --temperature, --top-p, --top-k, "
    "--threads, --ctx-size, --backend, --n-gpu-layers, --gpus, "
    "--draft-model, --draft-tokens, --file, --paged, --dflash-fusion, "
    "--mesh, --mesh-port, --pipe-head, --pipe-tail, --pipe-peer, --pipe-listen, "
    "--profile, --vision, --image"
)
