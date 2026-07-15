"""ctypes surface for the oxidize-c stable ABI (src/oxidize.h).

Loads liboxidize.so, mirrors the OxModelOptions / OxMetadata structs (with the
struct_size versioning field the ABI requires), and binds every ox_* entry
point with argtypes/restype so ctypes marshals correctly on 64-bit LP64.

The library is located, in order:
  1. $OXIDIZE_LIB (a full path to the .so), else
  2. liboxidize.so next to the repo root (../.. from this file), else
  3. the platform loader's search path (LD_LIBRARY_PATH etc.).
A clear RuntimeError is raised if none resolve.
"""
import ctypes
import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint64,
    c_void_p,
)


# ---- struct mirrors (must match src/oxidize.h field-for-field) --------------

class OxModelOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("ctx", c_size_t),
        ("threads", c_int),
        ("seed", c_uint64),
        ("kv_quant", c_int),
    ]


class OxMetadata(ctypes.Structure):
    _fields_ = [
        ("struct_size", c_size_t),
        ("arch", c_char_p),
        ("isa", c_char_p),
        ("vocab", c_size_t),
        ("ctx", c_size_t),
        ("n_tensors", c_size_t),
        ("n_kv", c_size_t),
    ]


# int (*OxTokenCb)(const char* piece, size_t len, void* user)
OxTokenCb = CFUNCTYPE(c_int, POINTER(c_char), c_size_t, c_void_p)


# ---- library location -------------------------------------------------------

def _lib_path():
    env = os.environ.get("OXIDIZE_LIB")
    if env:
        if not os.path.exists(env):
            raise RuntimeError(f"OXIDIZE_LIB={env!r} does not exist")
        return env
    # repo-root sibling: python/oxidize_c/_ffi.py -> ../../liboxidize.so
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "liboxidize.so"))
    if os.path.exists(cand):
        return cand
    return "liboxidize.so"  # let the loader search; error surfaces below


def _load():
    path = _lib_path()
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(
            f"could not load liboxidize.so ({path!r}): {e}. "
            "Build it with `make lib` in the oxidize-c dir, or set OXIDIZE_LIB "
            "to the .so path."
        ) from e


lib = _load()


# ---- signatures -------------------------------------------------------------

_ERRBUF = c_size_t(256)

lib.ox_model_open.argtypes = [
    POINTER(c_void_p), c_char_p, POINTER(OxModelOptions), c_char_p, c_size_t,
]
lib.ox_model_open.restype = c_int

lib.ox_model_close.argtypes = [c_void_p]
lib.ox_model_close.restype = None

lib.ox_metadata.argtypes = [c_void_p, POINTER(OxMetadata)]
lib.ox_metadata.restype = c_int

lib.ox_session_new.argtypes = [c_void_p]
lib.ox_session_new.restype = c_void_p

lib.ox_session_free.argtypes = [c_void_p]
lib.ox_session_free.restype = None

for _name, _ty in [
    ("ox_session_set_temperature", c_float),
    ("ox_session_set_top_p", c_float),
    ("ox_session_set_min_p", c_float),
    ("ox_session_set_repeat_penalty", c_float),
    ("ox_session_set_frequency_penalty", c_float),
    ("ox_session_set_presence_penalty", c_float),
]:
    getattr(lib, _name).argtypes = [c_void_p, c_float]
    getattr(lib, _name).restype = None

lib.ox_session_set_top_k.argtypes = [c_void_p, c_int]
lib.ox_session_set_top_k.restype = None
lib.ox_session_set_seed.argtypes = [c_void_p, c_uint64]
lib.ox_session_set_seed.restype = None

lib.ox_generate.argtypes = [
    c_void_p, c_char_p, c_int, OxTokenCb, c_void_p, c_char_p, c_size_t,
]
lib.ox_generate.restype = c_int

lib.ox_version.argtypes = []
lib.ox_version.restype = c_char_p
lib.ox_isa.argtypes = []
lib.ox_isa.restype = c_char_p
