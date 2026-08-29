#!/usr/bin/env python3
"""Generate oxidize-c/src/compute/quant_tables.h from the Rust source tables.

This is a one-time porting helper. The Rust sources are the authoritative
reference (oxidize-core/src/compute/quantization.rs and the
quantization/{iq_grids.rs,*_grid_fragment.rs} files), themselves transcribed
verbatim from ggml-common.h. The C header preserves the exact bytes so the
AL/IQ/NVFP4 dequant paths produce bit-exact parity with Rust
(VAL-QUANT-006/007/009/016).

Run: python3 scripts/gen_quant_tables.py
Output: src/compute/quant_tables.h
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST_COMPUTE = ROOT.parent / "oxidize-core" / "src" / "compute"
OUT = ROOT / "src" / "compute" / "quant_tables.h"

BANNER = (
    "/* Auto-generated from oxidize-core Rust lookup tables. Bit-exact port of "
    "oxidize-core/src/compute/quantization{.rs,/iq_grids.rs,/iq1s_grid_fragment.rs,"
    "/iq2s_grid_fragment.rs,/iq2xs_grid_fragment.rs}. Source of truth: ggml-common.h "
    "(ggml-org/llama.cpp). Do not hand-edit. Regenerate via scripts/gen_quant_tables.py. */"
)


def extract_array(text: str, name: str, ty: str) -> str:
    """Extract `pub ... const NAME: [T; N] = [ ... ];` body as a single string."""
    # Match the entire array definition. We allow `pub`, `pub(crate)`, `pub(super)`,
    # `static` or `const`, and `[T; N]` or `[T; N]` types.
    pat = re.compile(
        r"(?:pub(?:\([^)]*\))?\s*)?(?:const|static)\s+" + re.escape(name) + r"\s*:\s*\["
        + re.escape(ty) + r"\s*;\s*\d+\s*\]\s*=\s*\[(.*?)\];",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"could not find array {name}: [{ty}; ...]")
    return m.group(1)


def normalize_nums(body: str) -> list[str]:
    without_comments = re.sub(r"//.*?$|/\*.*?\*/", "", body, flags=re.MULTILINE | re.DOTALL)
    number = re.compile(
        r"(?<![\w.])(-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?))"
        r"(?:_[iuf]\d+|[iuf]\d+)?(?![\w.])"
    )
    return [match.group(1) for match in number.finditer(without_comments)]


def fmt_c_array(name: str, ctype: str, tokens: list[str], per_line: int = 8) -> str:
    lines = []
    for i in range(0, len(tokens), per_line):
        chunk = ", ".join(tokens[i : i + per_line])
        lines.append("    " + chunk + ("," if i + per_line < len(tokens) else ""))
    body = "\n".join(lines)
    return f"static const {ctype} {name}[{len(tokens)}] = {{\n{body}\n}};"


def main() -> None:
    quant_rs = (RUST_COMPUTE / "quantization.rs").read_text()
    iq_grids_rs = (RUST_COMPUTE / "quantization" / "iq_grids.rs").read_text()
    iq2xs = (RUST_COMPUTE / "quantization" / "iq2xs_grid_fragment.rs").read_text()
    iq2s = (RUST_COMPUTE / "quantization" / "iq2s_grid_fragment.rs").read_text()
    iq1s = (RUST_COMPUTE / "quantization" / "iq1s_grid_fragment.rs").read_text()

    blocks: list[str] = []
    blocks.append(BANNER)
    blocks.append('#ifndef OXIDIZE_QUANT_TABLES_H')
    blocks.append('#define OXIDIZE_QUANT_TABLES_H')
    blocks.append('')
    blocks.append('#include <stdint.h>')
    blocks.append('')
    # KVALUES_IQ4NL — 16 i8
    blocks.append(fmt_c_array('KVALUES_IQ4NL', 'int8_t',
                              normalize_nums(extract_array(quant_rs, 'KVALUES_IQ4NL', 'i8'))))
    blocks.append('')
    # E2M1_DOUBLED_VALUES — 16 f32
    blocks.append(fmt_c_array('E2M1_DOUBLED_VALUES', 'float',
                              normalize_nums(extract_array(quant_rs, 'E2M1_DOUBLED_VALUES', 'f32'))))
    blocks.append('')
    # IQ3S_GRID — 512 u32
    blocks.append(fmt_c_array('IQ3S_GRID', 'uint32_t',
                              normalize_nums(extract_array(quant_rs, 'IQ3S_GRID', 'u32'))))
    blocks.append('')
    # KMASK_IQ2XS — 8 u8
    blocks.append(fmt_c_array('KMASK_IQ2XS', 'uint8_t',
                              normalize_nums(extract_array(iq_grids_rs, 'KMASK_IQ2XS', 'u8'))))
    blocks.append('')
    # KSIGNS_IQ2XS — 128 u8
    blocks.append(fmt_c_array('KSIGNS_IQ2XS', 'uint8_t',
                              normalize_nums(extract_array(iq_grids_rs, 'KSIGNS_IQ2XS', 'u8'))))
    blocks.append('')
    # IQ2XXS_GRID — 256 u64
    blocks.append(fmt_c_array('IQ2XXS_GRID', 'uint64_t',
                              normalize_nums(extract_array(iq_grids_rs, 'IQ2XXS_GRID', 'u64'))))
    blocks.append('')
    # IQ3XXS_GRID — 256 u32
    blocks.append(fmt_c_array('IQ3XXS_GRID', 'uint32_t',
                              normalize_nums(extract_array(iq_grids_rs, 'IQ3XXS_GRID', 'u32'))))
    blocks.append('')
    # IQ2XS_GRID — 512 u64
    blocks.append(fmt_c_array('IQ2XS_GRID', 'uint64_t',
                              normalize_nums(extract_array(iq2xs, 'IQ2XS_GRID', 'u64'))))
    blocks.append('')
    # IQ2S_GRID — 1024 u64
    blocks.append(fmt_c_array('IQ2S_GRID', 'uint64_t',
                              normalize_nums(extract_array(iq2s, 'IQ2S_GRID', 'u64'))))
    blocks.append('')
    # IQ1S_GRID — 2048 u64
    blocks.append(fmt_c_array('IQ1S_GRID', 'uint64_t',
                              normalize_nums(extract_array(iq1s, 'IQ1S_GRID', 'u64'))))
    blocks.append('')
    blocks.append('#endif /* OXIDIZE_QUANT_TABLES_H */')
    OUT.write_text("\n".join(blocks) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
