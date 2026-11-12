"""Gated DeltaNet chunked kernel tuned for Turing (T4, sm75).

Drop-in replacement for `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
(the signature subset transformers' qwen3_5 modeling uses). FLA's Triton
kernels are correct on sm75 but ~60x off roofline there (96 ms fwd / 301 ms
fwd+bwd per layer at T=4096, H=48). The arithmetic is tiny (~36 GFLOP fwd), so
this version:

  * keeps every intra-chunk quantity in large chunk-parallel batched GEMMs
    (batch = B*H*N, cuBLAS-friendly), all in fp32 for exact parity;
  * moves only the inherently sequential state recurrence into two small
    Triton kernels (`_scan_fwd`, `_scan_bwd`): one launch each instead of
    ~64 x 20 tiny launches, which matters because Kaggle gives 4 slow vCPUs;
  * has a hand-derived backward (checked against autograd of the naive
    recurrence to ~1e-7) that recomputes cheap intermediates.

Math (per head; chunk of C tokens, cumulative log-decay G, Gam_ij = exp(G_i-G_j), i>=j):
    A    = strict_lower((beta*k) k^T * Gam)       Tinv = (I + A)^-1
    u    = Tinv (beta*v)                          w    = Tinv (beta*k*exp(G))
    P    = tril(q k^T * Gam)                      (q pre-scaled)
    scan over chunks n, state S (K x V):
        v'_n    = u_n - w_n S_n
        S_{n+1} = exp(G_C) S_n + (k_n*exp(G_C-G))^T v'_n
    o_n = (q_n*exp(G)) S_n + P_n v'_n              (chunk-parallel, after the scan)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False

__all__ = ["chunk_gated_delta_rule", "naive_recurrent_gated_delta_rule"]

BV = 32  # V tile per program: S tile [K, BV] fp32 lives in registers
# Scan launch config (tuned on T4). F16=True feeds fp16 operands to the tensor
# cores (sm75 has no tf32) while keeping the running state/accumulators fp32.
# Measured at T=4096, H=48: fwd Triton 8.4 ms (0 spills) vs torch loop 12.8 ms;
# the bwd Triton scan still spills (best 29.7 ms) so the torch bmm loop (19 ms) is
# the default there (bwd=None).
SCAN_CFG = dict(fwd=dict(BV=16, num_warps=4, num_stages=1, F16=False),
                bwd=None)


if HAS_TRITON:
    @triton.jit
    def _scan_fwd(w, u, kd, dC, h0, Ss, vn, hT, N,
                  K: tl.constexpr, V: tl.constexpr, C: tl.constexpr, BV: tl.constexpr,
                  HAS_H0: tl.constexpr, F16: tl.constexpr):
        # State S [K, BV] is held as four [K/4, BV] register tiles so nothing
        # spills on Turing (255-reg cap); every operand tile is [C, K/4].
        DT = tl.float16 if F16 else tl.float32
        KQ: tl.constexpr = K // 4
        iv = tl.program_id(0)
        bh = tl.program_id(1)
        oc = tl.arange(0, C)
        ok = tl.arange(0, KQ)
        ov = iv * BV + tl.arange(0, BV)
        kv0 = ok[:, None] * V + ov[None, :]
        ck0 = oc[:, None] * K + ok[None, :]
        cv_off = oc[:, None] * V + ov[None, :]
        if HAS_H0:
            S0 = tl.load(h0 + bh * K * V + kv0 + 0 * KQ * V)
        else:
            S0 = tl.zeros([KQ, BV], dtype=tl.float32)
        if HAS_H0:
            S1 = tl.load(h0 + bh * K * V + kv0 + 1 * KQ * V)
        else:
            S1 = tl.zeros([KQ, BV], dtype=tl.float32)
        if HAS_H0:
            S2 = tl.load(h0 + bh * K * V + kv0 + 2 * KQ * V)
        else:
            S2 = tl.zeros([KQ, BV], dtype=tl.float32)
        if HAS_H0:
            S3 = tl.load(h0 + bh * K * V + kv0 + 3 * KQ * V)
        else:
            S3 = tl.zeros([KQ, BV], dtype=tl.float32)
        for n in range(N):
            cid = bh * N + n
            sb = Ss + cid * K * V
            wb = w + cid * C * K
            kb = kd + cid * C * K
            v_new = tl.load(u + cid * C * V + cv_off)
            tl.store(sb + kv0 + 0 * KQ * V, S0)
            v_new -= tl.dot(tl.load(wb + ck0 + 0 * KQ).to(DT), S0.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 1 * KQ * V, S1)
            v_new -= tl.dot(tl.load(wb + ck0 + 1 * KQ).to(DT), S1.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 2 * KQ * V, S2)
            v_new -= tl.dot(tl.load(wb + ck0 + 2 * KQ).to(DT), S2.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 3 * KQ * V, S3)
            v_new -= tl.dot(tl.load(wb + ck0 + 3 * KQ).to(DT), S3.to(DT), input_precision="ieee")
            tl.store(vn + cid * C * V + cv_off, v_new)
            d = tl.load(dC + cid)
            vd = v_new.to(DT)
            S0 = S0 * d + tl.dot(tl.trans(tl.load(kb + ck0 + 0 * KQ).to(DT)), vd, input_precision="ieee")
            S1 = S1 * d + tl.dot(tl.trans(tl.load(kb + ck0 + 1 * KQ).to(DT)), vd, input_precision="ieee")
            S2 = S2 * d + tl.dot(tl.trans(tl.load(kb + ck0 + 2 * KQ).to(DT)), vd, input_precision="ieee")
            S3 = S3 * d + tl.dot(tl.trans(tl.load(kb + ck0 + 3 * KQ).to(DT)), vd, input_precision="ieee")
        tl.store(hT + bh * K * V + kv0 + 0 * KQ * V, S0)
        tl.store(hT + bh * K * V + kv0 + 1 * KQ * V, S1)
        tl.store(hT + bh * K * V + kv0 + 2 * KQ * V, S2)
        tl.store(hT + bh * K * V + kv0 + 3 * KQ * V, S3)

    @triton.jit
    def _scan_bwd(kd, P, do, qd, w, dC, dhT, dSs, dvn, dh0, N,
                  K: tl.constexpr, V: tl.constexpr, C: tl.constexpr, BV: tl.constexpr,
                  F16: tl.constexpr):
        DT = tl.float16 if F16 else tl.float32
        KQ: tl.constexpr = K // 4
        iv = tl.program_id(0)
        bh = tl.program_id(1)
        oc = tl.arange(0, C)
        ok = tl.arange(0, KQ)
        ov = iv * BV + tl.arange(0, BV)
        kv0 = ok[:, None] * V + ov[None, :]
        ck0 = oc[:, None] * K + ok[None, :]
        kc0 = ok[:, None] + oc[None, :] * K              # same tile, loaded transposed [KQ, C]
        cv_off = oc[:, None] * V + ov[None, :]
        ccT_off = oc[:, None] + oc[None, :] * C           # P^T
        dS0 = tl.load(dhT + bh * K * V + kv0 + 0 * KQ * V)
        dS1 = tl.load(dhT + bh * K * V + kv0 + 1 * KQ * V)
        dS2 = tl.load(dhT + bh * K * V + kv0 + 2 * KQ * V)
        dS3 = tl.load(dhT + bh * K * V + kv0 + 3 * KQ * V)
        for it in range(N):
            n = N - 1 - it
            cid = bh * N + n
            sb = dSs + cid * K * V
            kb = kd + cid * C * K
            qb = qd + cid * C * K
            wb = w + cid * C * K
            do_t = tl.load(do + cid * C * V + cv_off).to(DT)
            dv = tl.dot(tl.load(P + cid * C * C + ccT_off).to(DT), do_t, input_precision="ieee")
            tl.store(sb + kv0 + 0 * KQ * V, dS0)               # grad wrt state entering chunk n+1
            dv += tl.dot(tl.load(kb + ck0 + 0 * KQ).to(DT), dS0.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 1 * KQ * V, dS1)               # grad wrt state entering chunk n+1
            dv += tl.dot(tl.load(kb + ck0 + 1 * KQ).to(DT), dS1.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 2 * KQ * V, dS2)               # grad wrt state entering chunk n+1
            dv += tl.dot(tl.load(kb + ck0 + 2 * KQ).to(DT), dS2.to(DT), input_precision="ieee")
            tl.store(sb + kv0 + 3 * KQ * V, dS3)               # grad wrt state entering chunk n+1
            dv += tl.dot(tl.load(kb + ck0 + 3 * KQ).to(DT), dS3.to(DT), input_precision="ieee")
            tl.store(dvn + cid * C * V + cv_off, dv)
            d = tl.load(dC + cid)
            dvd = dv.to(DT)
            dS0 = (dS0 * d + tl.dot(tl.load(qb + kc0 + 0 * KQ).to(DT), do_t, input_precision="ieee")
                   - tl.dot(tl.load(wb + kc0 + 0 * KQ).to(DT), dvd, input_precision="ieee"))
            dS1 = (dS1 * d + tl.dot(tl.load(qb + kc0 + 1 * KQ).to(DT), do_t, input_precision="ieee")
                   - tl.dot(tl.load(wb + kc0 + 1 * KQ).to(DT), dvd, input_precision="ieee"))
            dS2 = (dS2 * d + tl.dot(tl.load(qb + kc0 + 2 * KQ).to(DT), do_t, input_precision="ieee")
                   - tl.dot(tl.load(wb + kc0 + 2 * KQ).to(DT), dvd, input_precision="ieee"))
            dS3 = (dS3 * d + tl.dot(tl.load(qb + kc0 + 3 * KQ).to(DT), do_t, input_precision="ieee")
                   - tl.dot(tl.load(wb + kc0 + 3 * KQ).to(DT), dvd, input_precision="ieee"))
        tl.store(dh0 + bh * K * V + kv0 + 0 * KQ * V, dS0)
        tl.store(dh0 + bh * K * V + kv0 + 1 * KQ * V, dS1)
        tl.store(dh0 + bh * K * V + kv0 + 2 * KQ * V, dS2)
        tl.store(dh0 + bh * K * V + kv0 + 3 * KQ * V, dS3)


def _scan_fwd_torch(w, u, kd, dC, h0):
    BH, N, C, K = w.shape
    V = u.shape[-1]
    S = h0 if h0 is not None else w.new_zeros(BH, K, V)
    Ss = w.new_empty(BH, N, K, V)
    vn = torch.empty_like(u)
    for n in range(N):
        Ss[:, n] = S
        vn_n = torch.baddbmm(u[:, n], w[:, n], S, beta=1.0, alpha=-1.0)
        vn[:, n] = vn_n
        S = torch.baddbmm(S * dC[:, n, None, None], kd[:, n].transpose(-1, -2), vn_n)
    return Ss, vn, S


def _scan_bwd_torch(kd, P, do, qd, w, dC, dhT):
    BH, N, C, K = kd.shape
    dS = dhT.clone()
    dSs = torch.empty(BH, N, K, dS.shape[-1], device=kd.device, dtype=kd.dtype)
    dvn = torch.empty_like(do)
    for n in range(N - 1, -1, -1):
        dSs[:, n] = dS
        dv = torch.baddbmm(kd[:, n] @ dS, P[:, n].transpose(-1, -2), do[:, n])
        dvn[:, n] = dv
        dS = torch.baddbmm(dS * dC[:, n, None, None], qd[:, n].transpose(-1, -2), do[:, n])
        dS = torch.baddbmm(dS, w[:, n].transpose(-1, -2), dv, beta=1.0, alpha=-1.0)
    return dSs, dvn, dS


def _use_triton(x):
    return HAS_TRITON and x.is_cuda and x.shape[-1] % 64 == 0 and x.shape[-2] % 16 == 0


def scan_fwd(w, u, kd, dC, h0):
    if not (_use_triton(u) and w.shape[-1] % 64 == 0):
        return _scan_fwd_torch(w, u, kd, dC, h0)
    BH, N, C, K = w.shape
    V = u.shape[-1]
    Ss = w.new_empty(BH, N, K, V)
    vn = torch.empty_like(u)
    hT = w.new_empty(BH, K, V)
    cfg = SCAN_CFG["fwd"]
    _scan_fwd[(V // cfg["BV"], BH)](w, u, kd, dC, h0 if h0 is not None else hT, Ss, vn, hT, N,
                                    K=K, V=V, C=C, HAS_H0=h0 is not None, **cfg)
    return Ss, vn, hT


def scan_bwd(kd, P, do, qd, w, dC, dhT):
    if SCAN_CFG["bwd"] is None or not (_use_triton(do) and kd.shape[-1] % 64 == 0):
        return _scan_bwd_torch(kd, P, do, qd, w, dC, dhT)
    BH, N, C, K = kd.shape
    V = do.shape[-1]
    dSs = kd.new_empty(BH, N, K, V)
    dvn = torch.empty_like(do)
    dh0 = kd.new_empty(BH, K, V)
    cfg = SCAN_CFG["bwd"]
    _scan_bwd[(V // cfg["BV"], BH)](kd, P, do, qd, w, dC, dhT.contiguous(), dSs, dvn, dh0, N,
                                    K=K, V=V, C=C, **cfg)
    return dSs, dvn, dh0


if HAS_TRITON:
    @triton.jit
    def _to_chunks_fwd(x, y, rstd, T, H, Tp, sb, st, sh, sd,
                       D: tl.constexpr, BT: tl.constexpr, NORM: tl.constexpr):
        # x [B,T,H,D] (any strides, fp16/fp32) -> y [B*H, Tp, D] fp32 (+ L2 norm), zero padded
        bh = tl.program_id(0)
        it = tl.program_id(1)
        b = bh // H
        h = bh % H
        ot = it * BT + tl.arange(0, BT)
        od = tl.arange(0, D)
        m = (ot < T)[:, None]
        xv = tl.load(x + b * sb + h * sh + ot[:, None] * st + od[None, :] * sd, mask=m, other=0.0).to(tl.float32)
        if NORM:
            r = 1.0 / tl.sqrt(tl.sum(xv * xv, 1) + 1e-6)
            xv = xv * r[:, None]
            tl.store(rstd + bh * Tp + ot, r, mask=ot < Tp)
        tl.store(y + (bh * Tp + ot[:, None]) * D + od[None, :], xv, mask=(ot < Tp)[:, None])

    @triton.jit
    def _to_chunks_bwd(dy, y, rstd, dx, T, H, Tp, sb, st, sh, sd,
                       D: tl.constexpr, BT: tl.constexpr, NORM: tl.constexpr):
        bh = tl.program_id(0)
        it = tl.program_id(1)
        b = bh // H
        h = bh % H
        ot = it * BT + tl.arange(0, BT)
        od = tl.arange(0, D)
        m = (ot < T)[:, None]
        yo = (bh * Tp + ot[:, None]) * D + od[None, :]
        g = tl.load(dy + yo, mask=m, other=0.0)
        if NORM:
            yv = tl.load(y + yo, mask=m, other=0.0)
            r = tl.load(rstd + bh * Tp + ot, mask=ot < T, other=0.0)
            g = r[:, None] * (g - yv * tl.sum(yv * g, 1)[:, None])
        tl.store(dx + b * sb + h * sh + ot[:, None] * st + od[None, :] * sd,
                 g.to(dx.dtype.element_ty), mask=m)


class _ToChunks(torch.autograd.Function):
    """[B,T,H,D] -> [B*H, Tp, D] fp32 in one pass, optionally L2-normalizing rows."""

    @staticmethod
    def forward(ctx, x, Tp, norm):
        B, T, H, D = x.shape
        y = torch.empty(B * H, Tp, D, device=x.device, dtype=torch.float32)
        rstd = torch.empty(B * H, Tp, device=x.device, dtype=torch.float32) if norm else y
        BT = 32
        _to_chunks_fwd[(B * H, triton.cdiv(Tp, BT))](x, y, rstd, T, H, Tp, *x.stride(), D=D, BT=BT, NORM=norm)
        ctx.save_for_backward(y if norm else None, rstd if norm else None)
        ctx.meta = (x.shape, x.dtype, Tp, norm)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        (B, T, H, D), dtype, Tp, norm = ctx.meta
        dy = dy.contiguous()
        dx = torch.empty(B, T, H, D, device=dy.device, dtype=dtype)
        BT = 32
        _to_chunks_bwd[(B * H, triton.cdiv(T, BT))](dy, y if norm else dy, rstd if norm else dy, dx, T, H, Tp,
                                                    *dx.stride(), D=D, BT=BT, NORM=norm)
        return dx, None, None


def _to_chunks_torch(x, Tp, norm):
    x = x.transpose(1, 2).float()
    if norm:
        x = _L2Norm.apply(x)
    x = F.pad(x, (0, 0, 0, Tp - x.shape[2]))
    return x.reshape(-1, Tp, x.shape[-1])


def _rowdot(a, b):
    return torch.einsum("...ck,...ck->...c", a, b)


def _pre(q, k, g, beta, scale, C):
    """Chunk-parallel CxC intermediates. Per-row scalings (beta, exp(G), scale) are
    folded into the small CxC matrices / [C] vectors so the only [C, K]-sized
    elementwise tensor materialized is kd."""
    G = g.cumsum(-1)                                              # [BH,N,C]
    mask = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    Gam = (G.unsqueeze(-1) - G.unsqueeze(-2)).masked_fill_(~mask, float("-inf")).exp_()
    kT = k.transpose(-1, -2)
    KK = k @ kT
    A = (KK * Gam).mul_(beta.unsqueeze(-1)).tril_(-1)            # A_ij = b_i k_i.k_j Gam_ij
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    Tinv = torch.linalg.solve_triangular(A + eye, eye.expand_as(A), upper=False, unitriangular=True)
    eG = G.exp()
    Tb = Tinv * beta.unsqueeze(-2)                                # u = Tb v
    Tw = Tb * eG.unsqueeze(-2)                                    # w = Tw k
    QK = q @ kT
    P = (QK * Gam).mul_(scale)
    GC = G[..., -1:]
    ekd = (GC - G).exp_()
    kd = k * ekd.unsqueeze(-1)
    dC = GC.squeeze(-1).exp()
    qv = eG * scale
    return dict(Gam=Gam, KK=KK, A=A, Tinv=Tinv, eG=eG, Tb=Tb, Tw=Tw, P=P, ekd=ekd, kd=kd, dC=dC, qv=qv)


class _ChunkGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, C, h0):
        # q,k (L2-normalized, unscaled): [BH,N,C,K] fp32 ; v: [BH,N,C,V] ; g, beta: [BH,N,C]
        I = _pre(q, k, g, beta, scale, C)
        u = I["Tb"] @ v
        w = I["Tw"] @ k
        Ss, vn, S = scan_fwd(w, u, I["kd"], I["dC"].contiguous(), h0)
        o = (q @ Ss).mul_(I["qv"].unsqueeze(-1))
        o.view(-1, C, o.shape[-1]).baddbmm_(I["P"].view(-1, C, C), vn.view(-1, C, vn.shape[-1]))
        ctx.save_for_backward(q, k, v, g, beta, Ss, vn)
        ctx.scale, ctx.C, ctx.has_h0 = scale, C, h0 is not None
        return o, S

    @staticmethod
    def backward(ctx, do, dS_final):
        q, k, v, g, beta, Ss, vn = ctx.saved_tensors
        scale, C = ctx.scale, ctx.C
        BH, N, _, K = q.shape
        Vd = v.shape[-1]
        I = _pre(q, k, g, beta, scale, C)
        Gam, Tinv, Tb, Tw, P, eG, qv = (I[x] for x in ("Gam", "Tinv", "Tb", "Tw", "P", "eG", "qv"))
        w = Tw @ k
        do = do.contiguous()
        if dS_final is None:
            dS_final = q.new_zeros(BH, K, Vd)
        qd = q * qv.unsqueeze(-1)
        dSs, dvn, dh0 = scan_bwd(I["kd"], P.contiguous(), do, qd, w, I["dC"].contiguous(), dS_final)
        del qd, w
        SsT = Ss.transpose(-1, -2)
        flat = lambda x: x.view(-1, x.shape[-2], x.shape[-1])  # noqa: E731
        # o = diag(qv) q Ss + P vn
        dqS = do @ SsT                                            # [C,K]
        dP = do @ vn.transpose(-1, -2)                            # [C,C]
        dG = _rowdot(dqS, q).mul_(qv)
        dq = dqS.mul_(qv.unsqueeze(-1))
        # state update: S_{n+1} = dC S_n + kd^T vn ; vn = u - w S_n
        dkd = vn @ dSs.transpose(-1, -2)
        t2 = _rowdot(dkd, I["kd"])                                # kd = k * exp(G_C - G)
        ddC = torch.einsum("bnkv,bnkv->bn", dSs, Ss)
        dw = dvn @ SsT                                            # (sign folded below)
        # u = Tb v ; w = Tw k   (du = dvn, dw_true = -dw)
        dv = Tb.transpose(-1, -2) @ dvn
        dTb = dvn @ v.transpose(-1, -2)
        dTw = dw @ k.transpose(-1, -2)                            # = -dTw_true
        dk = dkd.mul_(I["ekd"].unsqueeze(-1))
        flat(dk).baddbmm_(flat(Tw.transpose(-1, -2)), flat(dw), alpha=-1.0)
        # Tb_ij = Tinv_ij b_j ; Tw_ij = Tinv_ij b_j eG_j
        dTx = dTb - dTw * eG.unsqueeze(-2)                        # dTb + dTw_true * eG_j
        dbeta = (dTx * Tinv).sum(-2)
        dG -= (dTw * Tw).sum(-2)
        TinvT = Tinv.transpose(-1, -2)
        dA = -(TinvT @ (dTx * beta.unsqueeze(-2)) @ TinvT).tril_(-1)
        # A_ij = b_i KK_ij Gam_ij ; P = s QK Gam
        dAb = dA * Gam
        dbeta += (dAb * I["KK"]).sum(-1)
        dKK = dAb.mul_(beta.unsqueeze(-1))
        dKK = dKK + dKK.transpose(-1, -2)
        dQK = (dP * Gam).mul_(scale)
        t = (dA * I["A"]).add_(dP * P)
        dG += t.sum(-1) - t.sum(-2)
        flat(dk).baddbmm_(flat(dKK), flat(k)).baddbmm_(flat(dQK.transpose(-1, -2)), flat(q))
        flat(dq).baddbmm_(flat(dQK), flat(k))
        # kd = k * exp(G_C - G) ; dC = exp(G_C)
        dG -= t2
        dG[..., -1] += t2.sum(-1) + ddC * I["dC"]
        dg = dG.flip(-1).cumsum(-1).flip(-1)
        return dq, dk, dv, dg, dbeta, None, None, (dh0 if ctx.has_h0 else None)


class _L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rstd = torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
        y = x * rstd
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        return rstd * (dy - y * (y * dy).sum(-1, keepdim=True))


def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False, cu_seqlens=None, chunk_size=64, **_):
    """q,k: [B,T,H,K], v: [B,T,H,V], g (log decay, <=0): [B,T,H], beta: [B,T,H]."""
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_t4: varlen (cu_seqlens) not supported; use padded batches")
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = v.dtype
    C = chunk_size
    if scale is None:
        scale = K ** -0.5
    pad = (C - T % C) % C
    Tp = T + pad
    N = Tp // C
    tc = _ToChunks.apply if (HAS_TRITON and q.is_cuda) else _to_chunks_torch
    qf = tc(q, Tp, use_qk_l2norm_in_kernel).view(B * H, N, C, K)
    kf = tc(k, Tp, use_qk_l2norm_in_kernel).view(B * H, N, C, K)
    vf = tc(v, Tp, False).view(B * H, N, C, V)
    gf = F.pad(g.transpose(1, 2).float(), (0, pad)).reshape(B * H, N, C)
    bf = F.pad(beta.transpose(1, 2).float(), (0, pad)).reshape(B * H, N, C)
    h0 = None if initial_state is None else initial_state.float().reshape(B * H, K, V).contiguous()
    o, S = _ChunkGDN.apply(qf, kf, vf, gf, bf, scale, C, h0)
    o = o.view(B, H, Tp, V)[:, :, :T].transpose(1, 2).to(dtype)
    S = S.reshape(B, H, K, V) if output_final_state else None
    return o, S


def naive_recurrent_gated_delta_rule(q, k, v, g, beta, scale=None, use_qk_l2norm_in_kernel=False):
    """Reference token-by-token recurrence in fp32 (for tests)."""
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    S = q.new_zeros(B, H, K, v.shape[-1])
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, (v[:, t] - pred) * beta[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, 1), S


def warmup(device, H=48, K=128, V=128, T=256, B=1):
    """JIT-compile every Triton kernel of the fwd+bwd path once (cold-cache compile is
    ~20 s on Kaggle's 4 vCPUs). Shapes only need to hit the same specializations as
    the real call: head dims, fp16, T a multiple of the chunk size."""
    dev = torch.device(device)
    q, k = (torch.randn(B, T, H, K, device=dev, dtype=torch.float16, requires_grad=True) for _ in range(2))
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.float16, requires_grad=True)
    g = (-torch.rand(B, T, H, device=dev)).requires_grad_(True)
    beta = torch.rand(B, T, H, device=dev, dtype=torch.float16).requires_grad_(True)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    o.float().sum().backward()
    torch.cuda.synchronize(dev)
