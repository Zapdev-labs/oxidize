"""Hugging Face GGUF resolver mirroring oxidize-golang/hf/hub.go."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REVISION = "main"


@dataclass
class ResolveOptions:
    repo: str
    revision: str = DEFAULT_REVISION
    filename: str = ""
    cache_dir: str = ""
    api_base: str = "https://huggingface.co"
    cdn_base: str = "https://huggingface.co"


def _default_cache_dir() -> Path:
    home = Path.home()
    return home / ".cache" / "oxidize" / "hf"


def _split_repo_and_file(repo: str, explicit_file: str) -> tuple[str, str]:
    repo = repo.strip()
    if not repo:
        raise ValueError("hf: empty repo")
    if explicit_file:
        return repo, explicit_file
    parts = repo.split("/")
    if len(parts) >= 3 and parts[-1].lower().endswith(".gguf"):
        return "/".join(parts[:-1]), parts[-1]
    return repo, ""


def list_gguf_files(
    repo: str,
    revision: str = DEFAULT_REVISION,
    api_base: str = "https://huggingface.co",
) -> list[str]:
    _ = revision
    url = f"{api_base.rstrip('/')}/api/models/{repo}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    names: list[str] = []
    for sibling in payload.get("siblings", []):
        name = sibling.get("rfilename", "")
        if name.lower().endswith(".gguf"):
            names.append(name)
    names.sort()
    return names


def _pick_single_gguf(repo: str, revision: str, api_base: str) -> str:
    names = list_gguf_files(repo, revision, api_base)
    if not names:
        raise ValueError(f"hf: repo {repo!r} has no .gguf files")
    if len(names) == 1:
        return names[0]
    lines = "\n".join(f"  {n}" for n in names[:25])
    if len(names) > 25:
        lines += "\n  ..."
    raise ValueError(
        f"hf: repo {repo!r} has multiple .gguf files; specify --file. Candidates:\n{lines}"
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=0) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def resolve_gguf(opts: ResolveOptions) -> str:
    repo, filename = _split_repo_and_file(opts.repo, opts.filename)
    if opts.filename:
        filename = opts.filename
    if not filename:
        filename = _pick_single_gguf(repo, opts.revision or DEFAULT_REVISION, opts.api_base)
    rev = opts.revision or DEFAULT_REVISION
    cache = Path(opts.cache_dir) if opts.cache_dir else _default_cache_dir()
    dest_dir = cache / repo.replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return str(dest)
    url = f"{opts.cdn_base.rstrip('/')}/{repo}/resolve/{rev}/{filename}"
    _download(url, dest)
    return str(dest)
