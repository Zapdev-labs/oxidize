#!/usr/bin/env python3
"""Remove banner-only comments and collapse C/C++ file-header blocks.

Keeps the first descriptive line of a file header. Deletes lines that are
only a box/banner (unicode rules, ASCII dashes, equals). Does not touch
comments that mention an invariant, VAL- id, or a negation.

Usage:
  python3 scripts/strip_noise_comments.py --dry-run
  python3 scripts/strip_noise_comments.py --write
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "target",
    "vendor",
    ".claire",
    "node_modules",
    "__pycache__",
    ".venv",
    "dist",
    "criterion",
}

BANNER_LINE = re.compile(
    r"""^
    \s*
    (?:
        /\*\s*[─━=\-*].*[─━=\-*]{8,}.*\*/ |
        /\*\s*[─━=\-*]{8,}.*\*/           |
        /\*\s*[─━]{2,}.*\*/               |
        //\s*[─━=\-*].*[─━=\-*]{8,}.*     |
        //\s*[─━=\-*]{8,}.*               |
        \#\s*[─━=\-*].*[─━=\-*]{8,}.*     |
        \#\s*[─━=\-*]{8,}.*
    )
    \s*$
    """,
    re.VERBOSE,
)

KEEP_WORDS = re.compile(
    r"VAL-|bit-exact|invariant|must not|don't|do not|\bUB\b|"
    r"not a true inverse|\bf64\b|\bmmap\b|endian|\bwhy\b|because |"
    r"otherwise |workaround|not SIMD|\bmutex\b|\bdeadlock\b",
    re.IGNORECASE,
)

EXTRA_KEEP = re.compile(r"VAL-|bit-exact|invariant", re.IGNORECASE)

KEEP_LICENSE = re.compile(r"Copyright|SPDX|License", re.IGNORECASE)

NUMBERED_STEP = re.compile(r"^\s*/\*\s*\d+\.\s+.+\*/\s*$")

FILE_TITLE = re.compile(r"^\s*/\* \S+\.\w+")
MASTER_REF = os.environ.get("OXIDIZE_STRIP_REF", "origin/master")

C_EXTS = {".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx"}
OTHER_EXTS = {".rs", ".go", ".py"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def comment_text_lines(block: list[str]) -> list[str]:
    texts = []
    for raw in block:
        s = raw.strip()
        if s in {"/*", "*/", "*"}:
            continue
        if s.startswith("/*"):
            s = s[2:]
        if s.endswith("*/"):
            s = s[:-2]
        s = s.lstrip("*").strip()
        if s:
            texts.append(s)
    return texts


HANG_END = re.compile(
    r"\b(?:the|a|an|of|and|to|for|with|from|by|as|into|on|in|or|is)\s*$",
    re.IGNORECASE,
)


def first_sentence(text: str) -> str:
    joined = " ".join(text.split())
    for match in re.finditer(r"\.(?:\s|$)", joined):
        end = match.start() + 1
        before = joined[:end]
        if re.search(r"\.\w{1,3}\.$", before):
            continue
        if end >= 12:
            return before
    return joined


def collapse_c_blocks(lines: list[str]) -> list[str]:
    """Collapse 4+ line /* */ blocks to one complete sentence, plus keep-word lines."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("/*") and "*/" not in lines[i]:
            j = i + 1
            while j < len(lines) and "*/" not in lines[j]:
                j += 1
            if j < len(lines) and (j - i + 1) >= 4:
                block = lines[i : j + 1]
                texts = comment_text_lines(block)
                joined = " ".join(texts)
                if KEEP_LICENSE.search(joined):
                    out.extend(block)
                    i = j + 1
                    continue
                if not texts:
                    i = j + 1
                    continue
                indent = lines[i][: len(lines[i]) - len(stripped)]
                sentence = first_sentence(joined)
                if HANG_END.search(sentence.rstrip(".")):
                    sentence = texts[0].rstrip(".,;:") + "."
                out.append(indent + "/* " + sentence + " */")
                for extra in texts[1:]:
                    t = extra.strip()
                    if EXTRA_KEEP.search(t) and t not in sentence:
                        out.append(indent + "/* " + first_sentence(t) + " */")
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out


def strip_banners(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if FILE_TITLE.match(line):
            out.append(line)
            continue
        if BANNER_LINE.match(line) and not KEEP_WORDS.search(line):
            continue
        if NUMBERED_STEP.match(line) and not KEEP_WORDS.search(line):
            continue
        out.append(line)
    return out


def collapse_blank_runs(lines: list[str]) -> list[str]:
    out = []
    blanks = 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
        out.append(line)
    while out and out[-1].strip() == "":
        out.pop()
    out.append("")
    return out


ONELINE = re.compile(r"^(\s*)/\*(.*?)\*/(.*)$")


def is_bad_oneline(inner: str) -> bool:
    s = inner.strip()
    if HANG_END.search(s):
        return True
    if " We public " in s or "defined, and" in s:
        return True
    if len(s) > 130 and "." not in s[:100]:
        return True
    if len(s) > 140 and re.search(r"\.\s+[a-z]", s) and "VAL-" not in s:
        return True
    return False


def git_show(path: Path, ref: str = MASTER_REF) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        cwd=ROOT,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def opening_title(source: str) -> str | None:
    lines = source.splitlines()
    if not lines:
        return None
    if not lines[0].lstrip().startswith("/*"):
        return None
    if "*/" in lines[0]:
        inner = lines[0].split("/*", 1)[1].rsplit("*/", 1)[0].strip()
        return first_sentence(inner)
    block = [lines[0]]
    for line in lines[1:]:
        block.append(line)
        if "*/" in line:
            break
    texts = comment_text_lines(block)
    if not texts:
        return None
    return first_sentence(" ".join(texts))


def is_truncated_comment(inner: str) -> bool:
    s = inner.strip()
    if s.endswith((",", "(", ";", "the", "a", "an", "of", "and", "to", "for")):
        return True
    if HANG_END.search(s) and len(s) > 55:
        return True
    return False


def drop_fragments(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        match = ONELINE.match(line)
        if not match:
            out.append(line)
            continue
        inner = match.group(2).strip()
        if FILE_TITLE.match(line) or EXTRA_KEEP.search(inner):
            out.append(line)
            continue
        if inner[:1].islower() and not re.match(r"^[a-z0-9_.]+\.[a-z]{1,3}\b", inner):
            if len(inner) > 50 or is_truncated_comment(inner):
                continue
        out.append(line)
    return out


def restore_title(working: str, master: str) -> str:
    title = opening_title(master)
    if not title:
        return working
    lines = working.splitlines()
    first = next((line for line in lines if line.strip()), "")
    if FILE_TITLE.match(first):
        return working
    if first.startswith("#ifndef") or first.startswith("#define") or first.startswith("#include") or first.startswith("#pragma"):
        return f"/* {title} */\n" + working
    return working


def complete_truncated(working: str, master: str) -> str:
    comments = re.findall(r"/\*.*?\*/", master, re.S)
    parsed = [(" ".join(comment_text_lines(c.splitlines())), c) for c in comments]
    out = []
    for line in working.splitlines():
        match = ONELINE.match(line)
        if not match:
            out.append(line)
            continue
        indent, inner, rest = match.group(1), match.group(2), match.group(3)
        if not is_truncated_comment(inner):
            out.append(line)
            continue
        prefix = inner.strip()[:32]
        sentence = None
        for joined, _comment in parsed:
            if prefix and joined.startswith(prefix.strip()[:20]):
                sentence = first_sentence(joined)
                break
        if sentence:
            out.append(f"{indent}/* {sentence} */{rest}")
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if working.endswith("\n") else "")


def repair_onelines(working: str, head: str) -> str:
    comments = re.findall(r"/\*.*?\*/", head, re.S)
    parsed = []
    for comment in comments:
        texts = comment_text_lines(comment.splitlines())
        parsed.append((texts, " ".join(texts)))
    out = []
    for line in working.splitlines():
        match = ONELINE.match(line)
        if not match:
            out.append(line)
            continue
        indent, inner, rest = match.group(1), match.group(2), match.group(3)
        if not is_bad_oneline(inner):
            out.append(line)
            continue
        prefix = inner.strip()[:40]
        sentence = None
        extras = []
        for texts, joined in parsed:
            if prefix and joined.startswith(prefix.strip()[:24]):
                sentence = first_sentence(joined)
                extras = [
                    first_sentence(t)
                    for t in texts[1:]
                    if EXTRA_KEEP.search(t) and t not in sentence
                ][:1]
                break
        if not sentence:
            out.append(line)
            continue
        out.append(f"{indent}/* {sentence} */{rest}")
        for extra in extras:
            out.append(f"{indent}/* {extra} */")
    return "\n".join(out) + ("\n" if working.endswith("\n") else "")


def collapse_slash_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("//"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("//"):
                j += 1
            block = lines[i:j]
            joined = "\n".join(block)
            keep = (
                KEEP_WORDS.search(joined)
                or any(
                    token in line
                    for line in block
                    for token in ("go:build", "go:generate", "nolint", "//export ", "eslint")
                )
                or any(
                    line.lstrip().startswith(("//!", "///", "//go:")) for line in block
                )
            )
            if (j - i) >= 4 and not keep:
                out.append(block[0])
            else:
                out.extend(block)
            i = j
            continue
        out.append(lines[i])
        i += 1
    return out


def transform(path: Path, text: str) -> str:
    lines = text.splitlines()
    ext = path.suffix.lower()
    if ext in C_EXTS:
        lines = collapse_c_blocks(lines)
    if ext == ".go":
        lines = collapse_slash_blocks(lines)
    lines = strip_banners(lines)
    lines = collapse_blank_runs(lines)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="rewrite mashed one-line C comments from origin/master first sentences",
    )
    parser.add_argument(
        "--polish",
        action="store_true",
        help="restore file titles, drop fragments, complete truncated comments",
    )
    args = parser.parse_args()
    if not args.write and not args.dry_run:
        args.dry_run = True
    removed = 0
    files = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in C_EXTS | OTHER_EXTS:
                continue
            path = Path(dirpath) / name
            if should_skip(path):
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new = original
            if args.polish and ext in C_EXTS:
                master = git_show(path)
                if master is not None:
                    new = restore_title(new, master)
                    new = complete_truncated(new, master)
                lines = drop_fragments(new.splitlines())
                new = "\n".join(lines)
                new = collapse_blank_runs(new.splitlines())
                new = "\n".join(new)
            elif args.repair and ext in C_EXTS:
                master = git_show(path)
                if master is not None:
                    new = repair_onelines(new, master)
                new = transform(path, new)
            else:
                new = transform(path, new)
            if new == original:
                continue
            delta = original.count("\n") - new.count("\n")
            files += 1
            removed += delta
            if args.dry_run:
                print(f"{delta:5} {path.relative_to(ROOT)}")
            else:
                path.write_text(new, encoding="utf-8")
    print(f"files {files} lines_removed {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
