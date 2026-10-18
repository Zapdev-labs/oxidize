#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
/* Darwin hides mkdtemp unless the default surface is re-enabled. */
#define _DARWIN_C_SOURCE
#endif
#include <criterion/criterion.h>
#include "oxidize/exl3_overlay.h"
#include "oxidize/exl3.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Append fmt to buf, asserting the result fits. Keeps header-building
 * chains from silently advancing `n` past the buffer on truncation. */
static void hcat(char *buf, size_t cap, size_t *n, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    int w = (*n < cap) ? vsnprintf(buf + *n, cap - *n, fmt, ap) : -1;
    va_end(ap);
    cr_assert_geq(w, 0);
    cr_assert_lt(*n + (size_t)w, cap);
    *n += (size_t)w;
}

#define D 128u
#define PACKED (8u * 8u * 32u * 2u)
static const char *keys[] = {
    "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj", "lm_head"
};

static OcWeightView *view(OcLlamaModel *m, size_t i)
{
    OcWeightView *v[] = {&m->layers[0].attn_q, &m->layers[0].attn_k,
        &m->layers[0].attn_v, &m->layers[0].attn_output,
        &m->layers[0].ffn_gate, &m->layers[0].ffn_up,
        &m->layers[0].ffn_down, &m->output};
    return v[i];
}

static void init_model(OcLlamaModel *m, bool quantized)
{
    memset(m, 0, sizeof(*m));
    m->arch = OC_ARCH_LLAMA;
    m->cfg = (OcLlamaConfig){.vocab_size=D, .n_embd=D, .n_layer=1,
        .n_head=1, .n_head_kv=1, .n_ff=D, .n_ctx=16, .head_dim=D,
        .kv_head_dim=D, .rope_theta=10000, .rms_norm_eps=1e-5f, .norm_scale=1};
    m->layers = calloc(1, sizeof(*m->layers));
    cr_assert_not_null(m->layers);
    m->final_norm = malloc(D * sizeof(float));
    m->layers[0].attn_norm = malloc(D * sizeof(float));
    m->layers[0].ffn_norm = malloc(D * sizeof(float));
    m->layers[0].use_rope = true;
    for (size_t i=0; i<D; i++) {
        m->final_norm[i]=1; m->layers[0].attn_norm[i]=1;
        m->layers[0].ffn_norm[i]=1;
    }
    float *emb = malloc(D * D * sizeof(float));
    for (size_t i=0; i<D*D; i++) emb[i] = sinf((float)i * 0.17f);
    m->tok_embeddings = (OcWeightView){.data=(void *)emb, .qtype=OC_QUANT_F32,
        .rows=D, .cols=D, .row_bytes=D*sizeof(float)};
    for (size_t i=0; i<8; i++) {
        size_t rb = quantized ? (D / 32 * 34) : D * sizeof(float);
        *view(m,i) = (OcWeightView){.data=calloc(D,rb),
            .qtype=quantized ? OC_QUANT_Q8_0 : OC_QUANT_F32,
            .rows=D, .cols=D, .row_bytes=rb};
        cr_assert_not_null(view(m,i)->data);
    }
}

static void free_model(OcLlamaModel *m)
{
    for (size_t i=0;i<8;i++) free((void *)view(m,i)->data);
    free((void *)m->tok_embeddings.data);
    oc_llama_free(m);
}

/* Native input-tile-major packed words with nonuniform F16 scales. */
static void fixture(char path[64], size_t groups, int malformed)
{
    strcpy(path, "/tmp/oxidize-exl3-XXXXXX");
    int fd = mkstemp(path); cr_assert_geq(fd, 0);
    FILE *f = fdopen(fd, "wb"); cr_assert_not_null(f);
    char header[8192]; size_t n=0, offset=0;
    size_t scale_bytes = malformed == 3 ? 16 : 256;
    header[n++]='{';
    for (size_t i=0;i<groups;i++) {
        const char *key = (malformed == 1 && i == groups-1) ? "unsupported" : keys[i];
        hcat(header,sizeof(header),&n,
            "%s\"%s.trellis\":{\"dtype\":\"I16\",\"shape\":[8,8,%u],\"data_offsets\":[%zu,%zu]},"
            "\"%s.%s\":{\"dtype\":\"%s\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]},"
            "\"%s.%s\":{\"dtype\":\"%s\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}",
            i ? "," : "", key, malformed == 2 ? 16u : 32u, offset,offset+PACKED,
            key,malformed == 3 ? "su" : "suh",malformed == 3 ? "I16" : "F16",
            scale_bytes/2,offset+PACKED,offset+PACKED+scale_bytes,
            key,malformed == 3 ? "sv" : malformed == 6 ? "missing" : "svh",
            malformed == 3 ? "I16" : "F16",scale_bytes/2,
            offset+PACKED+scale_bytes,offset+PACKED+scale_bytes*2);
        if (malformed == 4 || malformed == 5 || malformed == 8) {
            hcat(header,sizeof(header),&n,
                ",\"%s.%s\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[%zu,%zu]}",
                key,malformed == 5 ? "bias" : "mul1",
                malformed == 4 ? (size_t)1000000 : 0,
                malformed == 4 ? (size_t)1000004 : 4);
            if (malformed == 8)
                hcat(header,sizeof(header),&n,
                    ",\"%s.mcg\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}",key);
        }
        offset += PACKED+scale_bytes*2;
    }
    header[n++]='}';
    uint8_t le[8]; for (size_t i=0;i<8;i++) le[i]=(uint8_t)((uint64_t)n>>(i*8));
    cr_assert_eq(fwrite(le,1,8,f),8);
    cr_assert_eq(fwrite(header,1,n,f),n);
    for (size_t g=0;g<groups;g++) {
        for (size_t i=0;i<PACKED;i++) fputc((int)((i*17+g*23+31)&255),f);
        for (size_t i=0;i<scale_bytes/2;i++) {
            fputc(malformed == 3 ? 0x55 : 0,f);
            fputc(malformed == 3 ? 0x55 : malformed == 7 ? 0x7c : i%2 ? 0x3c : 0xbc,f);
        }
        for (size_t i=0;i<scale_bytes/2;i++) {
            fputc(malformed == 3 ? 0xaa : 0,f);
            fputc(malformed == 3 ? 0xaa : i%3 ? 0x20 : 0xa0,f);
        }
    }
    cr_assert_eq(fclose(f),0);
}

Test(exl3_overlay, malformed_is_transactional)
{
    OcLlamaModel m; init_model(&m,false);
    const int modes[] = {1,2,4,5,6,7,8};
    for (size_t i=0; i<sizeof(modes)/sizeof(modes[0]); i++) {
        int bad = modes[i];
        char path[64]; fixture(path,2,bad);
        cr_assert_neq(oc_llama_load_exl3(&m,path),OC_OK);
        cr_assert_null(m.exl3_overlay);
        for(size_t i=0;i<8;i++) cr_assert_null(view(&m,i)->exl3);
        unlink(path);
    }
    free_model(&m);
}

Test(exl3_overlay, lifecycle_and_dimensions)
{
    OcLlamaModel m; init_model(&m,false);
    char path[64]; fixture(path,8,0);
    m.live_sessions=1;
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_ERR_INVALID_ARG);
    m.live_sessions=0;
    m.layers[0].attn_q.cols=64;
    cr_assert_neq(oc_llama_load_exl3(&m,path),OC_OK);
    cr_assert_null(m.exl3_overlay);
    m.layers[0].attn_q.cols=D;
    m.cfg.num_experts=2;
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_ERR_MODEL);
    m.cfg.num_experts=0;
    m.cfg.tied_embeddings=true;
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_OK);
    cr_assert(!m.cfg.tied_embeddings);
    cr_assert_null(m.tok_embeddings.exl3);
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_ERR_INVALID_ARG);
    unlink(path); free_model(&m);
}

static void compare(const float *a, const float *b)
{
    for(size_t i=0;i<D;i++) {
        cr_assert(isfinite(a[i]) && isfinite(b[i]));
        cr_assert_float_eq(a[i],b[i],2e-4f + fabsf(b[i])*2e-4f,"logit %zu",i);
    }
}

static void inference_parity(bool quantized)
{
    OcLlamaModel m, reference;
    init_model(&m,quantized); init_model(&reference,false);
    char path[64]; fixture(path,8,0);
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_OK);
    unlink(path);
    for(size_t w=0;w<8;w++) {
        float in[D]={0}, out[D];
        float *matrix=(float *)(void *)view(&reference,w)->data;
        for(size_t j=0;j<D;j++) {
            in[j]=1;
            cr_assert_eq(oc_exl3_matvec(view(&m,w)->exl3,in,out),OC_OK);
            for(size_t i=0;i<D;i++) matrix[i*D+j]=out[i];
            in[j]=0;
        }
    }
    OcLlamaSession a,b;
    cr_assert_eq(oc_llama_session_init_kv(&m,&a,OC_KV_F32),OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&reference,&b,OC_KV_F32),OC_OK);
    uint32_t tokens[]={1,3,5}; float x[D],y[D];
    for(size_t i=0;i<3;i++) {
        cr_assert_eq(oc_llama_forward(&a,tokens[i],x),OC_OK);
        cr_assert_eq(oc_llama_forward(&b,tokens[i],y),OC_OK);
        compare(x,y);
    }
    oc_llama_session_reset(&a); oc_llama_session_reset(&b);
    cr_assert_eq(oc_llama_prefill(&a,tokens,3,3,x),OC_OK);
    cr_assert_eq(oc_llama_prefill(&b,tokens,3,3,y),OC_OK);
    compare(x,y);
    oc_llama_session_free(&a); oc_llama_session_free(&b);
    free_model(&m); free_model(&reference);
}

Test(exl3_overlay, forward_prefill_f32_base) { inference_parity(false); }
Test(exl3_overlay, forward_prefill_q8_base) { inference_parity(true); }

Test(exl3_overlay, legacy_signs_are_lsb_first)
{
    OcLlamaModel m; init_model(&m,false);
    char path[64]; fixture(path,1,3);
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_OK);
    const OcExl3Matrix *q = view(&m,0)->exl3;
    for (size_t i=0;i<D;i++) {
        cr_assert_eq(q->suh[i],i%2 ? 1.0f : -1.0f);
        cr_assert_eq(q->svh[i],i%2 ? -1.0f : 1.0f);
    }
    unlink(path); free_model(&m);
}

Test(exl3_overlay, compute_errors_reach_forward_and_prefill)
{
    OcLlamaModel m; init_model(&m,false);
    char path[64]; fixture(path,8,0);
    cr_assert_eq(oc_llama_load_exl3(&m,path),OC_OK);
    unlink(path);
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m,&s,OC_KV_F32),OC_OK);
    OcExl3Matrix *q = (OcExl3Matrix *)view(&m,0)->exl3;
    q->packed_size--;
    float out[D]; uint32_t tokens[]={1,2,3};
    cr_assert_eq(oc_llama_forward(&s,1,out),OC_ERR_FORMAT);
    oc_llama_session_reset(&s);
    cr_assert_eq(oc_llama_prefill(&s,tokens,3,3,out),OC_ERR_FORMAT);
    q->packed_size++;
    oc_llama_session_reset(&s);
    cr_assert_eq(oc_llama_forward(&s,1,out),OC_OK);
    oc_llama_session_free(&s); free_model(&m);
}

static void native_fixture(char dir[64], bool tied, int bad)
{
    strcpy(dir,"/tmp/oxidize-native-exl3-XXXXXX");
    cr_assert_not_null(mkdtemp(dir));
    char file[256], packed[64];
    fixture(packed,bad == 3 ? 1 : tied ? 7 : 8,0);
    snprintf(file,sizeof(file),"%s/packed.safetensors",dir);
    cr_assert_eq(rename(packed,file),0);
    snprintf(file,sizeof(file),"%s/config.json",dir);
    FILE *f=fopen(file,"w"); cr_assert_not_null(f);
    fprintf(f,"{\"model_type\":\"llama\",\"vocab_size\":128,\"hidden_size\":128,"
        "\"num_hidden_layers\":1,\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"intermediate_size\":128,\"max_position_embeddings\":16,\"rms_norm_eps\":0.00001,"
        "\"tie_word_embeddings\":%s}",tied ? "true" : "false"); fclose(f);
    snprintf(file,sizeof(file),"%s/tokenizer.json",dir);
    f=fopen(file,"w"); cr_assert_not_null(f);
    fprintf(f,"{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for(size_t i=0;i<D;i++) {
        if(i<2) fprintf(f,"%s\"%c\":%zu",i ? "," : "",i ? 'b' : 'a',i);
        else fprintf(f,",\"t%zu\":%zu",i,i);
    }
    fprintf(f,"},\"merges\":[]},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"use_regex\":false,\"add_prefix_space\":false},\"decoder\":{\"type\":\"ByteLevel\"}}"); fclose(f);
    snprintf(file,sizeof(file),"%s/plain.safetensors",dir);
    f=fopen(file,"wb"); cr_assert_not_null(f);
    const char *names[]={"model.embed_tokens.weight","model.norm.weight",
        "model.layers.0.input_layernorm.weight","model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.bias","model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "lm_head.weight"};
    char header[4096]; size_t n=0,off=0; header[n++]='{';
    size_t count = bad == 3 ? 14 : 7;
    for(size_t i=0;i<count;i++) {
        size_t width = i>=7 && i%3 ? 2 : 4;
        size_t bytes=(i==0 || i>=7 ? D*D : D)*width;
        hcat(header,sizeof(header),&n,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%s128],\"data_offsets\":[%zu,%zu]}",
            i ? "," : "",bad==1 && i==2 ? "missing_norm" : names[i],width==4 ? "F32" : i%3==1 ? "F16" : "BF16",i==0 || i>=7 ? "128," : "",off,off+bytes);
        off+=bytes;
    }
    header[n++]='}'; uint8_t le[8];
    for(size_t i=0;i<8;i++) le[i]=(uint8_t)((uint64_t)n>>(i*8));
    fwrite(le,1,8,f); fwrite(header,1,n,f);
    for(size_t j=0;j<count;j++) for(size_t i=0;i<(j==0 || j>=7 ? D*D : D);i++) {
        float v=j==0 ? sinf((float)i*.17f) : j<4 ? 1 : .001f*(float)(i%5);
        if(bad==2 && j==2 && i==0) v=NAN;
        if(j>=7 && j%3) { uint16_t half=j%3==1 ? 0x211f : 0x3c23; fwrite(&half,2,1,f); }
        else fwrite(&v,4,1,f);
    }
    fclose(f);
}

static void remove_native_fixture(const char *dir)
{
    const char *names[]={"packed.safetensors","plain.safetensors","config.json","tokenizer.json"};
    char path[256];
    for(size_t i=0;i<4;i++) { snprintf(path,sizeof(path),"%s/%s",dir,names[i]); unlink(path); }
    rmdir(dir);
}

Test(exl3_native, checkpoint_forward_prefill_and_tied_head)
{
    for(int mode=0;mode<3;mode++) {
        bool tied=mode==1;
        char dir[64]; native_fixture(dir,tied,mode==2 ? 3 : 0);
        OcLlamaModel m; cr_assert_eq(oc_llama_load_exl3_native(dir,&m),OC_OK);
        cr_assert_eq(m.cfg.tied_embeddings,(bool)tied);
        cr_assert_not_null(m.layers[0].attn_q_bias);
        cr_assert_not_null(m.layers[0].attn_q.exl3);
        cr_assert_eq(m.gguf.unified.tensor_count,0);
        OcLlamaSession a,b;
        cr_assert_eq(oc_llama_session_init_kv(&m,&a,OC_KV_F32),OC_OK);
        cr_assert_eq(oc_llama_session_init_kv(&m,&b,OC_KV_F32),OC_OK);
        uint32_t tokens[]={1,3,5}; float x[D],y[D];
        for(size_t i=0;i<3;i++) cr_assert_eq(oc_llama_forward(&a,tokens[i],x),OC_OK);
        cr_assert_eq(oc_llama_prefill(&b,tokens,3,3,y),OC_OK);
        compare(x,y);
        oc_llama_session_free(&a); oc_llama_session_free(&b);
        oc_llama_free(&m); oc_llama_free(&m);
        if(getenv("OC_KEEP_NATIVE_FIXTURE")) fprintf(stderr,"NATIVE_FIXTURE=%s\n",dir);
        else remove_native_fixture(dir);
    }
}

Test(exl3_native, malformed_checkpoint_is_freed)
{
    for(int bad=1;bad<=2;bad++) {
        char dir[64]; native_fixture(dir,false,bad);
        OcLlamaModel m; cr_assert_neq(oc_llama_load_exl3_native(dir,&m),OC_OK);
        cr_assert_null(m.exl3_overlay); cr_assert_null(m.layers);
        oc_llama_free(&m); remove_native_fixture(dir);
    }
}

Test(exl3_native, duplicate_shard_and_missing_projection)
{
    char dir[64],a[256],b[256]; native_fixture(dir,false,0);
    snprintf(a,sizeof(a),"%s/plain.safetensors",dir);
    snprintf(b,sizeof(b),"%s/duplicate.safetensors",dir);
    cr_assert_eq(link(a,b),0);
    OcLlamaModel m; cr_assert_neq(oc_llama_load_exl3_native(dir,&m),OC_OK);
    cr_assert_null(m.layers); unlink(b);
    snprintf(a,sizeof(a),"%s/packed.safetensors",dir); unlink(a);
    cr_assert_neq(oc_llama_load_exl3_native(dir,&m),OC_OK);
    cr_assert_null(m.layers); remove_native_fixture(dir);
}
