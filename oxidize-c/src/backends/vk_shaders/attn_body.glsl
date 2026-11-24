/* attn_body.glsl — attention for a tile of TQ query tokens and one KV head
 * (its grp query heads, QH = TQ * grp rows), online softmax over chunks of KC
 * keys; the includer defines TQ and KC. With TQ_FILL the tile holds
 * QHM / grp tokens, so that every one of the QHM rows is used, and TQ only
 * bounds it. Scores
 * carry the learned relative bias for distance < EXT; the per-head sink
 * seeds the softmax (m = sink, l = 1); the output is gated by sigmoid(g).
 *
 * Per chunk: the K/V rows and bias span prefetched into registers go to
 * shared memory and the next chunk's loads are issued; LPR lanes per query
 * row score four keys each and fold them into the row's running max / sum
 * with clustered subgroup ops; each thread then accumulates RPT rows x four
 * dims of P.V. head_dim 128, grp <= 4, n_rel <= 16.
 *
 * Split keys (nsplit > 0): workgroup z takes share s0 + z of the keys, the
 * absolute positions [256 (s0 + z), 256 (s0 + z + 1)), and writes its
 * unnormalized P.V rows, running max and sum to part; attn_comb.comp merges
 * the shares with the sink and gates. Chunks then start at absolute positions
 * and every row sees the same shares whatever the call's other rows, so a
 * row's result does not depend on the batch it runs in. */
#define SHARE 256u
#include "common.glsl"
#extension GL_KHR_shader_subgroup_clustered : require
#define HD 128u
#define LPR (KC / 4u)
#ifndef QHM
#define QHM (TQ * 4u)
#endif
#define RPT (QHM / 8u)
#define KPT (KC / 8u)
#define Q4 33u
#define DSPAN (TQ + KC)
#define RTN (16u * DSPAN)
#define RTPT ((RTN + 255u) / 256u)
layout(local_size_x = 256) in;
layout(push_constant) uniform P {
    uint64_t q, r, g, sink, K, V, relT, o, part;
    uint n, pos0, NH, NKV, NR, EXT, cap, swa, n_swa, nsplit, s0;
    float kscale;
} p;

shared vec4  qs[QHM * Q4];
shared vec4  Ks[KC * Q4];
shared vec4  Vs[KC * 32u];
shared float ps[QHM][KC + 1u];
shared float rr[QHM][16];
shared float rt[16][DSPAN];
shared float ms[QHM], ls[QHM], cs[QHM];

void main()
{
    const uint tid = gl_LocalInvocationID.x;
    const uint grp = p.NH / p.NKV;
#ifdef TQ_FILL
    const uint tq_n = QHM / grp;
#else
    const uint tq_n = TQ;
#endif
    const uint t0 = gl_WorkGroupID.x * tq_n, g = gl_WorkGroupID.y;
    const uint QH = tq_n * grp, QD = p.NH * HD, KV = p.NKV * HD;
    const uint nt = min(tq_n, p.n - t0);
    FV4 Qv = FV4(p.q), Kv = FV4(p.K), Vv = FV4(p.V);
    F32 RT = F32(p.relT);

    for (uint i = tid; i < QHM * 32u; i += 256u) {
        const uint qh = i >> 5, d4 = i & 31u, tq = qh / grp, hi = qh % grp;
        qs[qh * Q4 + d4] = qh < QH && tq < nt ? Qv.v[((t0 + tq) * QD + (g * grp + hi) * HD) / 4u + d4] : vec4(0.0);
    }
    for (uint i = tid; i < QHM * 16u; i += 256u) {
        const uint qh = i >> 4, k = i & 15u, tq = qh / grp, hi = qh % grp;
        rr[qh][k] = qh < QH && tq < nt && k < p.NR ? F32(p.r).v[((t0 + tq) * p.NH + g * grp + hi) * p.NR + k] : 0.0;
    }
    for (uint i = tid; i < QHM; i += 256u) {
        const uint tq = i / grp, hi = i % grp;
        ms[i] = p.nsplit > 0u ? -1e30 : (i < QH && tq < nt ? F32(p.sink).v[(t0 + tq) * p.NH + g * grp + hi] : 0.0);
        ls[i] = p.nsplit > 0u ? 0.0 : 1.0;
    }

    const uint pq_first = p.pos0 + t0, pq_last = p.pos0 + t0 + nt - 1u;
    const uint klo = p.nsplit > 0u ? p.s0 * SHARE : p.swa != 0u && pq_first + 1u > p.n_swa ? pq_first + 1u - p.n_swa : 0u;
    const uint nk = pq_last - klo + 1u, sp = gl_WorkGroupID.z;
    const uint jb = p.nsplit > 0u ? min(nk, sp * SHARE) : 0u, je = p.nsplit > 0u ? min(nk, jb + SHARE) : nk;

    const uint lk = tid >> 5, ld = tid & 31u;
    vec4 kreg[KPT], vreg[KPT];
    float rreg[RTPT];
    [[unroll]] for (uint i = 0; i < KPT; i++) {
        const uint j = jb + lk + 8u * i;
        kreg[i] = vec4(0.0); vreg[i] = vec4(0.0);
        if (j < je) {
            const uint pk = klo + j, slot = p.swa != 0u ? pk % p.cap : pk, b4 = (slot * KV + g * HD) / 4u + ld;
            kreg[i] = Kv.v[b4]; vreg[i] = Vv.v[b4];
        }
    }
    [[unroll]] for (uint i = 0; i < RTPT; i++) {
        const uint e = tid + 256u * i, k = e / DSPAN, c = e % DSPAN;
        const int dd = int(pq_first) - int(klo + jb + KC - 1u) + int(c);
        rreg[i] = e < RTN && k < p.NR && dd >= 0 && uint(dd) < p.EXT ? RT.v[k * p.EXT + uint(dd)] : 0.0;
    }

    const uint row = tid / LPR, c0 = (tid % LPR) * 4u;
    const bool row_ok = row < QH && row / grp < nt;
    const uint pq = p.pos0 + t0 + row / grp;
    const uint rg = tid >> 5, d4 = tid & 31u;
    vec4 acc[RPT];
    [[unroll]] for (uint m = 0; m < RPT; m++) acc[m] = vec4(0.0);

    for (uint j0 = jb; j0 < je; j0 += KC) {
        barrier();
        [[unroll]] for (uint i = 0; i < KPT; i++) {
            Ks[(lk + 8u * i) * Q4 + ld] = kreg[i];
            Vs[(lk + 8u * i) * 32u + ld] = vreg[i];
        }
        [[unroll]] for (uint i = 0; i < RTPT; i++) {
            const uint e = tid + 256u * i;
            if (e < RTN) rt[e / DSPAN][e % DSPAN] = rreg[i];
        }
        const int base = int(pq_first) - int(klo + j0 + KC - 1u);
        barrier();
        if (j0 + KC < je) {
            [[unroll]] for (uint i = 0; i < KPT; i++) {
                const uint j = j0 + KC + lk + 8u * i;
                kreg[i] = vec4(0.0); vreg[i] = vec4(0.0);
                if (j < je) {
                    const uint pk = klo + j, slot = p.swa != 0u ? pk % p.cap : pk, b4 = (slot * KV + g * HD) / 4u + ld;
                    kreg[i] = Kv.v[b4]; vreg[i] = Vv.v[b4];
                }
            }
            [[unroll]] for (uint i = 0; i < RTPT; i++) {
                const uint e = tid + 256u * i, k = e / DSPAN, c = e % DSPAN;
                const int dd = base - int(KC) + int(c);
                rreg[i] = e < RTN && k < p.NR && dd >= 0 && uint(dd) < p.EXT ? RT.v[k * p.EXT + uint(dd)] : 0.0;
            }
        }

        float s[4] = float[4](NEG_INF, NEG_INF, NEG_INF, NEG_INF);
        if (row_ok) {
            float a[4] = float[4](0.0, 0.0, 0.0, 0.0);
            for (uint x = 0; x < 32u; x++) {
                const vec4 qv = qs[row * Q4 + x];
                [[unroll]] for (uint i = 0; i < 4u; i++) a[i] += dot(qv, Ks[(c0 + i) * Q4 + x]);
            }
            [[unroll]] for (uint i = 0; i < 4u; i++) {
                const uint pk = klo + j0 + c0 + i;
                if (j0 + c0 + i < je && pk <= pq && (p.swa == 0u || pq - pk < p.n_swa)) {
                    s[i] = a[i] * p.kscale;
                    const uint dist = pq - pk;
                    if (dist < p.EXT) {
                        const uint c = uint(int(dist) - base);
                        float b = 0.0;
                        for (uint k = 0; k < p.NR; k++) b += rr[row][k] * rt[k][c];
                        s[i] += b;
                    }
                }
            }
        }
        if (row < QHM) {
            const float lm = max(max(s[0], s[1]), max(s[2], s[3]));
            const float mold = ms[row];
            const float mx = max(mold, subgroupClusteredMax(lm, LPR));
            float lsum = 0.0;
            [[unroll]] for (uint i = 0; i < 4u; i++) { s[i] = exp(s[i] - mx); ps[row][c0 + i] = s[i]; lsum += s[i]; }
            const float sum = subgroupClusteredAdd(lsum, LPR);
            if (tid % LPR == 0u && row_ok) {
                const float c = exp(mold - mx);
                cs[row] = c;
                ls[row] = ls[row] * c + sum;
                ms[row] = mx;
            }
        }
        barrier();

        if (rg * RPT < QH) {
            [[unroll]] for (uint m = 0; m < RPT; m++) acc[m] *= cs[rg * RPT + m];
            for (uint j = 0; j < KC; j++) {
                const vec4 vv = Vs[j * 32u + d4];
                [[unroll]] for (uint m = 0; m < RPT; m++) acc[m] += ps[rg * RPT + m][j] * vv;
            }
        }
    }

    if (p.nsplit > 0u) {
        const uint rec = QHM * 130u, base = ((gl_WorkGroupID.x * p.NKV + g) * p.nsplit + sp) * rec;
        F32 PT = F32(p.part);
        [[unroll]] for (uint m = 0; m < RPT; m++) {
            const uint qh = rg * RPT + m;
            if (qh < QH) FV4(p.part + uint64_t(base + qh * 128u) * 4u).v[d4] = acc[m];
        }
        if (tid < QHM) { PT.v[base + QHM * 128u + tid] = ms[tid]; PT.v[base + QHM * 129u + tid] = ls[tid]; }
        return;
    }
    FV4 Ov = FV4(p.o), Gv = FV4(p.g);
    [[unroll]] for (uint m = 0; m < RPT; m++) {
        const uint qh = rg * RPT + m, tq = qh / grp, hi = qh % grp;
        if (qh < QH && tq < nt) {
            const uint idx = ((t0 + tq) * QD + (g * grp + hi) * HD) / 4u + d4;
            const vec4 gv = Gv.v[idx];
            Ov.v[idx] = acc[m] / ls[qh] / (vec4(1.0) + exp(-gv));
        }
    }
}
