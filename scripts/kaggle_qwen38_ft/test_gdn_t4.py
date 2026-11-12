"""Correctness + speed test for gdn_t4 (run on a CUDA box: python test_gdn_t4.py)."""
import os, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_t4 import chunk_gated_delta_rule  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


def inputs(B, T, H, K, V, dtype, req=True):
    q = torch.randn(B, T, H, K, device=dev).to(dtype)
    k = torch.randn(B, T, H, K, device=dev).to(dtype)
    v = torch.randn(B, T, H, V, device=dev).to(dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=dev)) * 0.5      # <= 0, realistic range
    beta = torch.rand(B, T, H, device=dev).to(dtype)
    ts = [q, k, v, g, beta]
    return [t.requires_grad_(req) for t in ts]


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()


def check_grads(B=2, T=200, H=4, K=32, V=48):
    ins = inputs(B, T, H, K, V, torch.float32)
    h0 = torch.randn(B, H, K, V, device=dev, requires_grad=True)
    o, S = chunk_gated_delta_rule(*ins, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    do, dS = torch.randn_like(o), torch.randn_like(S)
    (o * do).sum().backward(retain_graph=True)
    (S * dS).sum().backward()
    g_mine = [t.grad.clone() for t in ins] + [h0.grad.clone()]
    for t in ins + [h0]:
        t.grad = None

    # reference: naive recurrence with initial state, via autograd
    def ref(q, k, v, g, beta, h0):
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        S = h0
        outs = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None, None]
            kt = k[:, t]
            pred = torch.einsum("bhk,bhkv->bhv", kt, S)
            S = S + torch.einsum("bhk,bhv->bhkv", kt, (v[:, t] - pred) * beta[:, t][..., None])
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * K ** -0.5, S))
        return torch.stack(outs, 1), S

    o2, S2 = ref(*ins, h0)
    ((o2 * do).sum() + (S2 * dS).sum()).backward()
    g_ref = [t.grad for t in ins] + [h0.grad]
    print("fwd o rel", rel(o, o2), "S rel", rel(S, S2))
    errs = {n: rel(a, b) for n, a, b in zip("q k v g beta h0".split(), g_mine, g_ref)}
    print("grad rel", errs)
    assert rel(o, o2) < 1e-4 and rel(S, S2) < 1e-4, "forward mismatch"
    assert max(errs.values()) < 1e-3, f"grad mismatch {errs}"


def bench(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3


def speed(T=4096, B=1, H=48, K=128, V=128):
    ins = inputs(B, T, H, K, V, torch.float16)
    impls = {"gdn_t4": chunk_gated_delta_rule}
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
        impls["fla"] = fla_chunk
    except Exception as e:  # noqa: BLE001
        print("fla unavailable:", e)
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as hf
        impls["hf_torch"] = getattr(hf, "__wrapped__", hf)
    except Exception as e:  # noqa: BLE001
        print("hf unavailable:", e)
    outs = {}
    for name, f in impls.items():
        def fw():
            with torch.no_grad():
                return f(*ins, use_qk_l2norm_in_kernel=True)[0]

        def fb():
            o, _ = f(*ins, use_qk_l2norm_in_kernel=True)
            o.float().square().mean().backward()
        outs[name] = fw()
        torch.cuda.reset_peak_memory_stats()
        t_fb = bench(fb)
        print(f"{name:9s} T={T} fwd {bench(fw):7.1f} ms  fwd+bwd {t_fb:7.1f} ms  "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if "fla" in outs:
        print("gdn_t4 vs fla fwd rel", rel(outs["gdn_t4"], outs["fla"]))


if __name__ == "__main__":
    import gdn_t4
    print("-- triton scan path"); check_grads(K=64, V=64)
    gdn_t4.HAS_TRITON = False
    print("-- torch scan path"); check_grads(K=32, V=48)
    gdn_t4.HAS_TRITON = True
    speed(int(os.environ.get("T", 4096)))
