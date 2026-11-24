/* moe_group.glsl — MoE work items, shared by group.comp and route_small.comp.
 * The includer declares shared uint cnt[256], first_[256] (pairs per expert
 * and, written here, each expert's first slot in the expert-sorted order).
 * Each expert's pairs are cut into pieces of up to 128: a piece of more than
 * MOE_SMALL (8) pairs becomes one item {expert, first pair, count, 0} of the
 * big list (nvfp4_mm), a smaller one an item of the small list (nvfp4_mv). ind receives four indirect dispatch commands: big gate_up,
 * big down, small gate_up, small down. Run by a single thread. */
#define MOE_SMALL 8u

void build_items(uint NE, uint64_t items, uint64_t items_s, uint64_t ind, uint gx_gu, uint gx_dn_b, uint gx_dn_s)
{
    uint acc = 0u, nb = 0u, ns = 0u;
    IV4 IT = IV4(items), IS = IV4(items_s);
    for (uint e = 0; e < NE; e++) {
        first_[e] = acc;
        for (uint c = 0; c < cnt[e]; c += 128u) {
            const uint pc = min(128u, cnt[e] - c);
            if (pc > MOE_SMALL) {
                IT.v[nb++] = ivec4(int(e), int(acc + c), int(pc), 0);
            } else {
                IS.v[ns++] = ivec4(int(e), int(acc + c), int(pc), 0);
            }
        }
        acc += cnt[e];
    }
    U32 I = U32(ind);
    I.v[0] = gx_gu;   I.v[1] = nb;  I.v[2] = 1u;
    I.v[3] = gx_dn_b; I.v[4] = nb;  I.v[5] = 1u;
    I.v[6] = gx_gu;   I.v[7] = ns;  I.v[8] = 1u;
    I.v[9] = gx_dn_s; I.v[10] = ns; I.v[11] = 1u;
}
