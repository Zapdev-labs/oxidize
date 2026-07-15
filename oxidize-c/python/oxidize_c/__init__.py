"""Pythonic bindings for oxidize-c over its stable C ABI (pure ctypes, no build).

    import oxidize_c
    with oxidize_c.OxidizeModel("model.gguf", ctx=4096, threads=0, seed=42) as m:
        print(m.metadata())
        for piece in m.generate("hello", max_tokens=64, temperature=0.7):
            print(piece, end="", flush=True)

    oxidize_c.version()  # -> "oxidize-c 0.1.0"
    oxidize_c.isa()      # -> "avx2" / "avx512+vnni" / ...

Generation streams: ox_generate runs on a worker thread and its C callback
pushes decoded UTF-8 pieces onto a queue that generate() yields from. Breaking
out of the loop early signals the callback to stop the C-side generation
cleanly (the callback returns non-zero, which the ABI treats as a clean stop).
"""
import codecs
import ctypes
import queue
import threading

from ._ffi import OxMetadata, OxModelOptions, OxTokenCb, lib

__all__ = ["OxidizeModel", "OxidizeError", "version", "isa"]

_ERRLEN = 256


class OxidizeError(RuntimeError):
    """Raised when a C ABI call returns -1; carries the C error string."""


def version():
    """Library version string, e.g. 'oxidize-c 0.1.0'."""
    return lib.ox_version().decode("utf-8", "replace")


def isa():
    """Active kernel ISA string, e.g. 'avx2'."""
    return lib.ox_isa().decode("utf-8", "replace")


def _errbuf():
    return ctypes.create_string_buffer(_ERRLEN)


class OxidizeModel:
    """One loaded GGUF. Open once; heavy. Use as a context manager to close it.

    Wraps a single shared OxSession (the ABI serializes generations on a model
    anyway), so calls to generate() form a multi-turn conversation.
    """

    def __init__(self, path, ctx=0, threads=0, seed=0, kv_quant=False):
        opts = OxModelOptions(
            struct_size=ctypes.sizeof(OxModelOptions),
            ctx=ctx,
            threads=threads,
            seed=seed,
            kv_quant=1 if kv_quant else 0,
        )
        handle = ctypes.c_void_p()
        err = _errbuf()
        rc = lib.ox_model_open(
            ctypes.byref(handle),
            str(path).encode("utf-8"),
            ctypes.byref(opts),
            err,
            _ERRLEN,
        )
        if rc != 0:
            raise OxidizeError(err.value.decode("utf-8", "replace") or "ox_model_open failed")
        self._model = handle
        self._session = lib.ox_session_new(handle)
        if not self._session:
            lib.ox_model_close(self._model)
            self._model = None
            raise OxidizeError("ox_session_new failed (allocation)")

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        if getattr(self, "_session", None):
            lib.ox_session_free(self._session)
            self._session = None
        if getattr(self, "_model", None):
            lib.ox_model_close(self._model)
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        self.close()

    # -- metadata -----------------------------------------------------------

    def metadata(self):
        """Return an --inspect-style dict (arch, isa, vocab, ctx, n_tensors, n_kv)."""
        md = OxMetadata(struct_size=ctypes.sizeof(OxMetadata))
        if lib.ox_metadata(self._model, ctypes.byref(md)) != 0:
            raise OxidizeError("ox_metadata failed")
        return {
            "arch": md.arch.decode("utf-8", "replace") if md.arch else None,
            "isa": md.isa.decode("utf-8", "replace") if md.isa else None,
            "vocab": md.vocab,
            "ctx": md.ctx,
            "n_tensors": md.n_tensors,
            "n_kv": md.n_kv,
        }

    # -- sampling -----------------------------------------------------------

    def _apply_sampler(self, temperature, top_k, top_p, min_p,
                       repeat_penalty, frequency_penalty, presence_penalty, seed):
        s = self._session
        if temperature is not None:
            lib.ox_session_set_temperature(s, temperature)
        if top_k is not None:
            lib.ox_session_set_top_k(s, top_k)
        if top_p is not None:
            lib.ox_session_set_top_p(s, top_p)
        if min_p is not None:
            lib.ox_session_set_min_p(s, min_p)
        if repeat_penalty is not None:
            lib.ox_session_set_repeat_penalty(s, repeat_penalty)
        if frequency_penalty is not None:
            lib.ox_session_set_frequency_penalty(s, frequency_penalty)
        if presence_penalty is not None:
            lib.ox_session_set_presence_penalty(s, presence_penalty)
        if seed is not None:
            lib.ox_session_set_seed(s, seed)

    # -- generation ---------------------------------------------------------

    def generate(self, prompt, max_tokens=128, temperature=None, top_k=None,
                 top_p=None, min_p=None, repeat_penalty=None,
                 frequency_penalty=None, presence_penalty=None, seed=None):
        """Yield decoded UTF-8 text pieces as the model generates them.

        Any sampler argument left None keeps the session's current setting.
        Breaking out of the returned generator stops C-side generation cleanly.
        Raises OxidizeError if the C generate call fails.
        """
        if prompt is None:
            raise OxidizeError("prompt must not be None")
        self._apply_sampler(temperature, top_k, top_p, min_p, repeat_penalty,
                            frequency_penalty, presence_penalty, seed)

        q = queue.Queue(maxsize=64)
        stop = threading.Event()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        _SENTINEL = object()

        def _cb(piece, length, _user):
            if stop.is_set():
                return 1  # clean stop
            text = decoder.decode(ctypes.string_at(piece, length))
            if text:
                q.put(text)
            return 0

        cb = OxTokenCb(_cb)  # keep a ref so it isn't GC'd during the C call
        err = _errbuf()
        result = {}

        def _run():
            rc = lib.ox_generate(
                self._session, prompt.encode("utf-8"), max_tokens, cb, None,
                err, _ERRLEN,
            )
            result["rc"] = rc
            result["err"] = err.value.decode("utf-8", "replace")
            tail = decoder.decode(b"", final=True)
            if tail and not stop.is_set():
                q.put(tail)
            q.put(_SENTINEL)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            stop.set()
            # If we broke early the worker may be parked on a full-queue put();
            # drain so its next callback sees `stop` and returns 1 (clean stop).
            while worker.is_alive():
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            worker.join()

        if result.get("rc", 0) != 0:
            raise OxidizeError(result.get("err") or "ox_generate failed")
