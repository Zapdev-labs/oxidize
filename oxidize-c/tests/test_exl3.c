#include <criterion/criterion.h>
#include "oxidize/exl3.h"
#include "oxidize/parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Exhaustive float32-bit FNV hashes from the pinned independent NumPy
 * trellis.decode oracle; includes every binary16 tie-rounding case. */
Test(exl3, codebooks_exhaustive_golden)
{
    const uint64_t expected[] = {UINT64_C(0x4eca7e4f6c57c325), UINT64_C(0xcf4a97c6f335c325), UINT64_C(0xdb1dd42b22dba325)};
    for (unsigned cb = 0; cb < 3; ++cb) {
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned i = 0; i < 65536; ++i) {
            float f = oc_exl3_codebook_value(cb, (uint16_t)i);
            uint32_t u; memcpy(&u, &f, sizeof u);
            hash = (hash ^ u) * UINT64_C(1099511628211);
        }
        cr_assert_eq(hash, expected[cb], "codebook %u RNE mismatch", cb);
    }
    cr_assert(isnan(oc_exl3_codebook_value(3, 0)), "invalid codebook");
}
static void fill_bytes(uint8_t *p, size_t size)
{
    uint32_t state = 0x592781a3;
    for (size_t i = 0; i < size; ++i) { state = state * 1664525u + 1013904223u; p[i] = (uint8_t)(state >> 24); }
}
static void reference_tile(const OcExl3Matrix *m, size_t kt, size_t nt, float *out)
{
    const uint8_t *p = m->packed + (kt*(m->output_dim/16)+nt)*16*m->bits_twice;
    unsigned end = 0, ring = 128*m->bits_twice;
    for (unsigned i = 0; i < 256; ++i) {
        end += m->bits_twice/2 + ((m->bits_twice & 1) && (i & 1));
        unsigned idx = 0;
        for (unsigned bit = 0; bit < 16; ++bit) {
            unsigned pos = (end+ring-16+bit)%ring;
            unsigned byte = (pos/32)*4 + 3-(pos%32)/8;
            idx = idx*2 + ((p[byte] >> (7-pos%8)) & 1);
        }
        unsigned t = i/8, j = i%8;
        out[((t%4)*2+(j&1)+(j&2 ? 8:0))*16+t/4+(j&4 ? 8:0)] = oc_exl3_codebook_value(m->codebook, (uint16_t)idx);
    }
}
Test(exl3, packed_all_rates_wrap_and_order)
{
    uint8_t packed[16384]; float scale[128], got[256], expected[256];
    fill_bytes(packed, sizeof packed);
    for (unsigned i = 0; i < 128; ++i) scale[i] = 1;
    for (unsigned cb = 0; cb < 3; ++cb)
        for (unsigned b = 2; b <= 16; ++b) {
            if ((b&1) && (cb != 2 || b > 7)) continue;
            OcExl3Matrix m = {128,128,packed,64*16*b,b,cb,scale,scale,NULL};
            for (size_t kt = 0; kt < 8; ++kt)
                for (size_t nt = 0; nt < 8; ++nt) {
                    reference_tile(&m, kt, nt, expected);
                    cr_assert_eq(oc_exl3_decode_tile(&m,kt,nt,got),OC_OK,"decode");
                    cr_assert_eq(memcmp(got,expected,sizeof got),0,"rate %u cb %u tile %zu,%zu",b,cb,kt,nt);
                }
        }
}
Test(exl3, malformed)
{
    uint8_t p[2048] = {0}; float s[128];
    for (unsigned i=0;i<128;++i) s[i]=1;
    OcExl3Matrix m = {128,128,p,sizeof p,2,0,s,s,NULL}, bad;
    cr_assert_eq(oc_exl3_validate(&m),OC_OK,"valid");
    cr_assert_neq(oc_exl3_validate(NULL),OC_OK,"null");
#define BAD(field,value) do { bad=m; bad.field=(value); cr_assert_neq(oc_exl3_validate(&bad),OC_OK,"invalid " #field); } while (0)
    BAD(input_dim,0); BAD(output_dim,127); BAD(packed,NULL); BAD(packed_size,2047);
    BAD(packed_size,2049); BAD(bits_twice,0); BAD(bits_twice,3); BAD(bits_twice,18);
    BAD(codebook,3); BAD(suh,NULL); BAD(svh,NULL); BAD(input_dim,SIZE_MAX-127);
    bad=m; bad.codebook=2; bad.bits_twice=9;
    cr_assert_neq(oc_exl3_validate(&bad),OC_OK,"unsupported fractional");
    s[9]=NAN; cr_assert_neq(oc_exl3_validate(&m),OC_OK,"nonfinite scale"); s[9]=1;
    float bias[128]={0}; bad=m; bad.bias=bias; bias[0]=INFINITY;
    cr_assert_neq(oc_exl3_validate(&bad),OC_OK,"nonfinite bias");
    float out[256];
    cr_assert_neq(oc_exl3_decode_tile(&m,8,0,out),OC_OK,"tile bounds");
    cr_assert_neq(oc_exl3_matmul(&m,s,128,out,128,0),OC_OK,"zero batch");
    cr_assert_neq(oc_exl3_matmul(&m,s,127,out,128,1),OC_OK,"input stride");
    cr_assert_neq(oc_exl3_matmul(&m,s,128,out,127,1),OC_OK,"output stride");
    cr_assert_neq(oc_exl3_matmul(&m,s,SIZE_MAX,out,128,2),OC_OK,"stride overflow");
#undef BAD
}
static double hsign(unsigned i, unsigned j)
{
    unsigned v=i&j, parity=0; while(v) { parity^=v&1; v>>=1; } return parity ? -1:1;
}
Test(exl3, batch_dense_reference_and_parallel)
{
    enum { K=128, N=1024, B=5, IS=131, OS=1027 };
    uint8_t *p=malloc((K/16)*(N/16)*48);
    float *got=malloc(B*OS*sizeof(float)), *serial=malloc(B*OS*sizeof(float));
    cr_assert(p && got && serial,"allocate");
    float in[B*IS], suh[K], svh[N], bias[N];
    fill_bytes(p,(K/16)*(N/16)*48);
    for(unsigned k=0;k<K;++k) suh[k]=(k&1 ? -1:1)*(0.5f+(float)(k%7)/8);
    for(unsigned n=0;n<N;++n) {svh[n]=(n&1 ? -1:1)*0.25f; bias[n]=(float)(n%9)/32;}
    for(unsigned i=0;i<B*IS;++i) in[i]=(float)((int)(i%17)-8)/16;
    for(unsigned i=0;i<B*OS;++i) got[i]=serial[i]=12345;
    OcExl3Matrix m={K,N,p,(K/16)*(N/16)*48,3,2,suh,svh,bias};
    cr_assert_eq(oc_parallel_set_threads(1),OC_OK,"serial pool");
    cr_assert_eq(oc_exl3_matmul(&m,in,IS,serial,OS,B),OC_OK,"serial product");
    cr_assert_eq(oc_parallel_set_threads(4),OC_OK,"parallel pool");
    cr_assert_eq(oc_exl3_matmul(&m,in,IS,got,OS,B),OC_OK,"parallel product");
    cr_assert_eq(memcmp(got,serial,B*OS*sizeof(float)),0,"thread deterministic");
    for(unsigned row=0;row<B;++row) {
        double x[K]={0}, acc[N]={0}; float tile[256];
        for(unsigned i=0;i<K;++i) for(unsigned j=0;j<K;++j)
            x[i]+=(double)in[row*IS+j]*suh[j]*hsign(i,j)/sqrt(128.0);
        for(unsigned kt=0;kt<K/16;++kt) for(unsigned nt=0;nt<N/16;++nt) {
            reference_tile(&m,kt,nt,tile);
            for(unsigned k=0;k<16;++k) for(unsigned n=0;n<16;++n) acc[nt*16+n]+=x[kt*16+k]*tile[k*16+n];
        }
        for(unsigned n=0;n<N;++n) {
            double sum=0;
            for(unsigned j=0;j<128;++j) sum+=acc[(n/128)*128+j]*hsign(n%128,j)/sqrt(128.0);
            double expected=sum*svh[n]+bias[n];
            cr_assert(fabs(got[row*OS+n]-expected)<2e-5*(1+fabs(expected)),"dense mismatch row %u col %u",row,n);
        }
        for(unsigned n=N;n<OS;++n) cr_assert_eq(got[row*OS+n],12345,"padding unchanged");
    }
    cr_assert_eq(oc_exl3_matvec(&m,in,got),OC_OK,"matvec");
    cr_assert_eq(memcmp(got,serial,N*sizeof(float)),0,"matvec equals first batch row");
    oc_parallel_shutdown(); free(p); free(got); free(serial);
}
