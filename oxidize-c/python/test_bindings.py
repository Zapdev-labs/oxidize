"""Tests for the oxidize_c ctypes bindings.

Run (from this python/ dir, with liboxidize.so already built via `make lib`):
    pytest test_bindings.py -v
or without pytest:
    python test_bindings.py

Covers everything the ABI exposes: version()/isa() strings, error propagation
on a bad path, metadata, and full streaming generate() against a tiny but
runnable llama GGUF synthesized here (mirroring tests/test_abi.c's builder, so
no external model file is needed).
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oxidize_c  # noqa: E402


# ---- minimal GGUF v3 writer (port of tests/test_abi.c write_model) ----------

def _u32(v): return struct.pack("<I", v)
def _u64(v): return struct.pack("<Q", v)
def _f32(v): return struct.pack("<f", v)
def _gstr(s):
    b = s.encode("utf-8")
    return _u64(len(b)) + b


class _KV:
    def __init__(self):
        self.buf = bytearray()
        self.n = 0

    def u32(self, k, v):
        self.buf += _gstr(k) + _u32(4) + _u32(v); self.n += 1

    def f32(self, k, v):
        self.buf += _gstr(k) + _u32(6) + _f32(v); self.n += 1

    def s(self, k, v):
        self.buf += _gstr(k) + _u32(8) + _gstr(v); self.n += 1

    def sarr(self, k, items):
        self.buf += _gstr(k) + _u32(9) + _u32(8) + _u64(len(items))
        for it in items:
            self.buf += _gstr(it)
        self.n += 1


_rs = [1234]
def _rndf():
    _rs[0] = (_rs[0] * 1103515245 + 12345) & 0xFFFFFFFF
    return (((_rs[0] >> 16) & 0xFFFF) - 32768) * (1.0 / 32768.0)


def _write_tiny_model(path):
    H, NL, NH, KVH, HD, FF, V = 64, 3, 4, 2, 16, 128, 32
    toks = ["<unk>", "<s>", "</s>"] + [chr(ord("a") + i) for i in range(29)]
    assert len(toks) == V

    kv = _KV()
    kv.s("general.architecture", "llama")
    kv.u32("llama.embedding_length", H)
    kv.u32("llama.block_count", NL)
    kv.u32("llama.attention.head_count", NH)
    kv.u32("llama.attention.head_count_kv", KVH)
    kv.u32("llama.feed_forward_length", FF)
    kv.u32("llama.context_length", 64)
    kv.f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    kv.f32("llama.rope.freq_base", 1e4)
    kv.u32("llama.rope.dimension_count", HD)
    kv.sarr("tokenizer.ggml.tokens", toks)
    kv.u32("tokenizer.ggml.unknown_token_id", 0)
    kv.u32("tokenizer.ggml.bos_token_id", 1)
    kv.u32("tokenizer.ggml.add_bos_token", 0)
    kv.u32("tokenizer.ggml.add_space_prefix", 0)

    # (name, rows, cols, centre); rows==0 => 1-D of `cols` values
    tensors = [
        ("token_embd.weight", V, H, 0.0),
        ("output.weight", V, H, 0.0),
        ("output_norm.weight", 0, H, 1.0),
    ]
    for l in range(NL):
        p = f"blk.{l}."
        tensors += [
            (p + "attn_q.weight", NH * HD, H, 0.0),
            (p + "attn_k.weight", KVH * HD, H, 0.0),
            (p + "attn_v.weight", KVH * HD, H, 0.0),
            (p + "attn_output.weight", H, NH * HD, 0.0),
            (p + "ffn_gate.weight", FF, H, 0.0),
            (p + "ffn_up.weight", FF, H, 0.0),
            (p + "ffn_down.weight", H, FF, 0.0),
            (p + "attn_norm.weight", 0, H, 1.0),
            (p + "ffn_norm.weight", 0, H, 1.0),
        ]

    z = bytearray()
    z += b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(kv.n) + kv.buf

    off = 0
    for name, rows, cols, _c in tensors:
        nvals = rows * cols if rows else cols
        z += _gstr(name)
        z += _u32(2 if rows else 1)
        z += _u64(cols)
        if rows:
            z += _u64(rows)
        z += _u32(0)  # F32 tensor
        z += _u64(off)
        off += (nvals * 4 + 31) & ~31
    while len(z) % 32:
        z += b"\x00"

    for _name, rows, cols, centre in tensors:
        nvals = rows * cols if rows else cols
        for _ in range(nvals):
            z += _f32(centre + 0.1 * _rndf())
        while len(z) % 32:
            z += b"\x00"

    with open(path, "wb") as f:
        f.write(z)


# ---- tests ------------------------------------------------------------------

def test_version_and_isa():
    v = oxidize_c.version()
    a = oxidize_c.isa()
    assert isinstance(v, str) and v, v
    assert isinstance(a, str) and a, a
    assert "oxidize" in v.lower()


def test_bad_path_raises():
    try:
        oxidize_c.OxidizeModel("/no/such/model.gguf")
    except oxidize_c.OxidizeError as e:
        assert str(e), "error must carry a message"
    else:
        raise AssertionError("opening a bad path should raise")


def test_metadata_and_generate():
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
        path = tf.name
    try:
        _write_tiny_model(path)
        # struct_size is set correctly by the binding, so open must succeed
        # (a wrong struct_size is rejected by the C side before load).
        with oxidize_c.OxidizeModel(path, ctx=0, threads=0, seed=42) as m:
            md = m.metadata()
            assert md["arch"] == "llama", md
            assert md["vocab"] == 32, md
            assert md["ctx"] == 64, md
            assert md["n_tensors"] == 3 + 3 * 9, md
            assert md["isa"], md

            pieces = list(m.generate("hello", max_tokens=8, temperature=0.0,
                                     repeat_penalty=1.1))
            assert len(pieces) > 0, "callback/generator produced no pieces"
            assert all(isinstance(p, str) for p in pieces)

            # multi-turn continuation on the same session
            more = list(m.generate("hi", max_tokens=4, temperature=0.0))
            assert len(more) > 0

            # early break must not hang (drains + clean-stops the worker)
            gen = m.generate("hello", max_tokens=64, temperature=0.0)
            first = next(gen)
            assert isinstance(first, str)
            gen.close()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_version_and_isa()
    print("ok version/isa:", oxidize_c.version(), "|", oxidize_c.isa())
    test_bad_path_raises()
    print("ok bad-path raises")
    test_metadata_and_generate()
    print("ok metadata + streaming generate + multi-turn + early-break")
    print("all binding tests passed")
