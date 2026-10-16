#include "oxidize/hf_metadata.h"
#include "oxidize/json.h"
#include "oxidize/log.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HF_JSON_LIMIT (128u * 1024u * 1024u)
static bool nullish(const OcJson *v) { return !v || v->type == OC_JSON_NULL; }
static bool eq(const OcJson *v, const char *s) { return v && v->type == OC_JSON_STRING && !strcmp(v->string,s); }
static bool flag(const OcJson *o,const char *key,bool fallback,bool *out) {
    const OcJson *v=oc_json_get(o,key);if(!v){*out=fallback;return true;}if(v->type!=OC_JSON_BOOL)return false;*out=v->boolean;return true;
}
static bool uint_field(const OcJson *o,const char *key,uint32_t fallback,uint32_t *out) {
    const OcJson *v=oc_json_get(o,key);if(!v){*out=fallback;return true;}return oc_json_u32(v,out);
}
static bool float_field(const OcJson *o,const char *key,float fallback,float *out) {
    const OcJson *v=oc_json_get(o,key);if(!v){*out=fallback;return true;}
    if(v->type!=OC_JSON_NUMBER||v->number<=0||v->number>FLT_MAX)return false;
    *out=(float)v->number;return *out>0;
}
static OcError read_json(const char *dir,const char *name,bool optional,OcArena *arena,OcJson **out) {
    *out=NULL;size_t n=strlen(dir)+strlen(name)+2;char *path=malloc(n);if(!path)return OC_ERR_OOM;
    snprintf(path,n,"%s/%s",dir,name);FILE *fp=fopen(path,"rb");free(path);
    if(!fp)return optional&&errno==ENOENT?OC_OK:OC_ERR_IO;
    if(fseek(fp,0,SEEK_END)) {fclose(fp);return OC_ERR_IO;}long size=ftell(fp);
    if(size<0||(unsigned long)size>HF_JSON_LIMIT||fseek(fp,0,SEEK_SET)){fclose(fp);return OC_ERR_FORMAT;}
    char *buf=malloc((size_t)size+1);if(!buf){fclose(fp);return OC_ERR_OOM;}
    bool ok=fread(buf,1,(size_t)size,fp)==(size_t)size;fclose(fp);
    OcError e=ok?oc_json_parse(buf,(size_t)size,arena,out):OC_ERR_IO;free(buf);
    if(e==OC_OK&&(*out)->type!=OC_JSON_OBJECT)e=OC_ERR_FORMAT;
    return e;
}
static OcGgufMetadataValue *entry(OcGgufFile *m,const char *key,OcGgufMetadataType type) {
    if(m->metadata_kv_count>=32)return NULL;
    OcGgufMetadataKV *kv=&m->metadata[m->metadata_kv_count++];kv->key=oc_arena_dup(m->arena,key);if(!kv->key)return NULL;kv->value.type=type;return &kv->value;
}
static bool setstr(OcGgufFile *m,const char *key,const char *str) {
    OcGgufMetadataValue *v=entry(m,key,OC_GGUF_MT_STRING);if(!v)return false;
    v->v.str.data=oc_arena_dup(m->arena,str);v->v.str.len=strlen(str);return v->v.str.data!=NULL;
}
static bool setu32(OcGgufFile *m,const char *key,uint32_t n) {OcGgufMetadataValue *v=entry(m,key,OC_GGUF_MT_UINT32);if(!v)return false;v->v.u32=n;return true;}
static OcGgufMetadataValue *array(OcGgufFile *m,const char *key,OcGgufMetadataType type,size_t n) {
    if(n>SIZE_MAX/sizeof(OcGgufMetadataValue))return NULL;
    OcGgufMetadataValue *v=entry(m,key,OC_GGUF_MT_ARRAY);if(!v)return NULL;
    v->v.arr.elem_type=type;v->v.arr.len=n;v->v.arr.values=oc_arena_alloc_bytes(m->arena,n*sizeof(OcGgufMetadataValue));
    if(!v->v.arr.values)return NULL;
    memset(v->v.arr.values,0,n*sizeof(OcGgufMetadataValue));for(size_t i=0;i<n;i++)v->v.arr.values[i].type=type;return v->v.arr.values;
}
static bool token(OcGgufFile *m,OcGgufMetadataValue *v,const char *text) {
    v->v.str.data=oc_arena_dup(m->arena,text);v->v.str.len=strlen(text);return v->v.str.data!=NULL;
}
/* qwen4_exp_text (Qwen3.8-Flash-Next family, e.g. CYBER-FROST): the real
 * fields live in the nested "text_config" object and the semantics differ
 * from every dense arch, so map them here rather than through the generic
 * llama path. Behavior in llama.c keys off cfg->is_qwen4exp; the arch enum
 * stays OC_ARCH_QWEN. Mirrors the GGUF qwen4exp metadata contract:
 *   linear_*            -> ssm.*             (gated delta-net geometry)
 *   hc_count/hc_lowrank -> hyper_connection.*
 *   indexer_*           -> attention.indexer.*
 *   layer_types         -> attention.recurrent_layers
 *   ple_layer_ids       -> ple.layers        (1-based in HF, 0-based here)
 *   rope_parameters     -> rope.dimension_count/sections (interleaved MRoPE)
 * PLE hash constants and ple_embd_dim are not in config.json — they live in
 * ngram_embedding.safetensors and are filled in by the native loader. */
static OcError config_load_qwen4exp(const OcJson *j,OcLlamaConfig *c) {
    const OcJson *t=oc_json_get(j,"text_config");
    const OcJson *tmt=oc_json_get(t,"model_type");
    if(!t||(tmt&&!eq(tmt,"qwen4_exp_text")))return OC_ERR_MODEL;
    if(!uint_field(t,"vocab_size",0,&c->vocab_size)||!uint_field(t,"hidden_size",0,&c->n_embd)||
       !uint_field(t,"num_hidden_layers",0,&c->n_layer)||!uint_field(t,"num_attention_heads",0,&c->n_head)||
       !uint_field(t,"num_key_value_heads",c->n_head,&c->n_head_kv)||!uint_field(t,"head_dim",0,&c->head_dim)||
       !uint_field(t,"max_position_embeddings",4096,&c->n_ctx)||!float_field(t,"rms_norm_eps",1e-5f,&c->rms_norm_eps))return OC_ERR_FORMAT;
    if(!c->vocab_size||!c->n_embd||!c->n_layer||!c->n_head||!c->n_head_kv||c->n_head%c->n_head_kv||
       !c->head_dim||c->head_dim%2||!c->n_ctx||c->n_head_kv>c->n_head)return OC_ERR_MODEL;
    const OcJson *act=oc_json_get(t,"hidden_act");if(act&&!eq(act,"silu"))return OC_ERR_MODEL;
    bool ab=false;if(!flag(t,"attention_bias",false,&ab)||ab)return OC_ERR_MODEL;
    c->is_qwen4exp=true;c->rope_norm_pairs=true;c->attn_out_gate=false;
    /* MoE: every layer is routed (top-k of num_experts + shared expert). */
    if(!uint_field(t,"num_experts",0,&c->num_experts)||!uint_field(t,"num_experts_per_tok",0,&c->num_experts_per_tok)||
       !uint_field(t,"moe_intermediate_size",0,&c->expert_intermediate_size)||
       !uint_field(t,"shared_expert_intermediate_size",0,&c->shared_expert_intermediate_size))return OC_ERR_FORMAT;
    if(!c->num_experts||!c->num_experts_per_tok||c->num_experts_per_tok>c->num_experts||!c->expert_intermediate_size)return OC_ERR_MODEL;
    c->n_ff=c->expert_intermediate_size;c->moe_layer_start=0;
    /* RoPE: interleaved MRoPE over head_dim*partial_rotary_factor dims. */
    const OcJson *rp=oc_json_get(t,"rope_parameters");
    const OcJson *tt=rp?rp:t;
    float theta;if(!float_field(tt,"rope_theta",10000,&theta))return OC_ERR_MODEL;c->rope_theta=theta;
    float partial=1;const OcJson *pv=oc_json_get(tt,"partial_rotary_factor");
    if(pv&&(pv->type!=OC_JSON_NUMBER||pv->number<=0||pv->number>1))return OC_ERR_MODEL;
    if(pv)partial=(float)pv->number;
    c->rope_dim=(uint32_t)((uint64_t)c->head_dim*partial);if(!c->rope_dim||c->rope_dim>c->head_dim)return OC_ERR_MODEL;
    c->kv_head_dim=c->head_dim;
    const OcJson *sec=rp?oc_json_get(rp,"mrope_section"):NULL;
    if(sec&&sec->type==OC_JSON_ARRAY) {
        size_t i=0;for(const OcJson *v=sec->child;v&&i<4;v=v->next,i++)
            if(v->type!=OC_JSON_NUMBER||v->number<0||!oc_json_u32(v,&c->rope_sections[i]))return OC_ERR_FORMAT;
    }
    /* Recurrent/full layer kinds, hyper-connections, QSA indexer. */
    const OcJson *lt=oc_json_get(t,"layer_types");
    if(!lt||lt->type!=OC_JSON_ARRAY||lt->count!=c->n_layer)return OC_ERR_MODEL;
    c->recurrent_layers=calloc(c->n_layer,sizeof(uint8_t));if(!c->recurrent_layers)return OC_ERR_OOM;
    size_t li=0;for(const OcJson *v=lt->child;v;v=v->next,li++)
        if(eq(v,"linear_attention"))c->recurrent_layers[li]=1;
        else if(!eq(v,"full_attention"))return OC_ERR_MODEL;
    if(!uint_field(t,"full_attention_interval",1,&c->full_attention_interval)||!c->full_attention_interval)return OC_ERR_FORMAT;
    if(!uint_field(t,"hc_count",0,&c->hc_mult)||!uint_field(t,"hc_lowrank",0,&c->hc_low_rank)||
       !c->hc_mult||c->hc_mult>8u||!c->hc_low_rank)return OC_ERR_MODEL;
    if(!uint_field(t,"indexer_n_heads",0,&c->indexer_n_head)||!uint_field(t,"indexer_head_dim",0,&c->indexer_head_size)||
       !uint_field(t,"indexer_budget",0,&c->indexer_top_k)||!uint_field(t,"indexer_compress_ratio",1,&c->indexer_kpool))return OC_ERR_FORMAT;
    if(!c->indexer_n_head||!c->indexer_head_size||!c->indexer_top_k)return OC_ERR_MODEL;
    /* Gated delta-net geometry. */
    if(!uint_field(t,"linear_conv_kernel_dim",0,&c->ssm_conv_kernel)||
       !uint_field(t,"linear_key_head_dim",0,&c->ssm_state_size)||
       !uint_field(t,"linear_num_key_heads",0,&c->ssm_group_count)||
       !uint_field(t,"linear_num_value_heads",0,&c->ssm_value_heads)||
       !uint_field(t,"linear_value_head_dim",0,&c->ssm_inner_size))return OC_ERR_FORMAT;
    c->ssm_inner_size*=c->ssm_value_heads;
    if(c->ssm_conv_kernel<2||!c->ssm_state_size||!c->ssm_group_count||!c->ssm_value_heads||
       c->ssm_inner_size%c->ssm_value_heads||c->ssm_value_heads%c->ssm_group_count)return OC_ERR_MODEL;
    /* MTP draft head: HF counts it separately from num_hidden_layers. */
    if(!uint_field(t,"mtp_num_hidden_layers",0,&c->nextn_predict_layers))return OC_ERR_FORMAT;
    if(c->nextn_predict_layers>=c->n_layer)return OC_ERR_MODEL;
    /* PLE: layer ids are 1-based in HF. Hash constants are read from
     * ngram_embedding.safetensors by the native loader; if that file is
     * absent it disables ple_enabled and logs. */
    const OcJson *pl=oc_json_get(t,"ple_layer_ids");
    if(pl&&pl->type==OC_JSON_ARRAY&&pl->count) {
        if(pl->count!=1)return OC_ERR_MODEL;
        uint32_t id;if(!oc_json_u32(pl->child,&id)||!id||id-1>=c->n_layer)return OC_ERR_MODEL;
        c->ple_layer=id-1;
        if(!uint_field(t,"ngram_size",0,&c->ple_ngram)||!uint_field(t,"heads_per_ngram",0,&c->ple_heads_per_ngram)||
           !uint_field(t,"ple_conv_kernel_size",0,&c->ple_conv_kernel))return OC_ERR_FORMAT;
        c->ple_n_heads=c->ple_ngram>0?(c->ple_ngram-1u)*c->ple_heads_per_ngram:0;
        if(c->ple_ngram<2||c->ple_ngram>OC_Q4E_PLE_MAX_NGRAM||!c->ple_n_heads||
           c->ple_n_heads>OC_Q4E_PLE_MAX_HEADS||c->ple_conv_kernel<2)return OC_ERR_MODEL;
        c->ple_eos=0;const OcJson *ev=oc_json_get(t,"eos_token_id");
        if(ev&&ev->type==OC_JSON_NUMBER)oc_json_u32(ev,&c->ple_eos);
        c->ple_img=0;if(oc_json_u32(oc_json_get(j,"image_token_id"),&c->ple_img)) {}
        c->ple_enabled=true;
    }
    bool tied=false;if(!flag(t,"tie_word_embeddings",false,&tied))return OC_ERR_FORMAT;c->tied_embeddings=tied;
    c->norm_scale=1;c->expert_weights_scale=1;c->sliding_window_pattern=1;c->norm_groups=1;
    c->head_dim_swa=c->head_dim;c->n_head_kv_swa=c->n_head_kv;c->rope_dim_swa=c->rope_dim;c->rope_theta_swa=c->rope_theta;
    return OC_OK;
}
static OcError config_load(const OcJson *j,OcLlamaConfig *c,OcModelArchitecture *arch) {
    const OcJson *type=oc_json_get(j,"model_type");
    if(eq(type,"qwen4_exp")||eq(type,"qwen4_exp_text")||eq(oc_json_get(oc_json_get(j,"text_config"),"model_type"),"qwen4_exp_text")) {
        *arch=OC_ARCH_QWEN;
        return config_load_qwen4exp(j,c);
    }
    if(eq(type,"llama"))*arch=OC_ARCH_LLAMA;else if(eq(type,"mistral"))*arch=OC_ARCH_MISTRAL;else if(eq(type,"qwen2"))*arch=OC_ARCH_QWEN;else return OC_ERR_MODEL;
    if(!uint_field(j,"vocab_size",0,&c->vocab_size)||!uint_field(j,"hidden_size",0,&c->n_embd)||!uint_field(j,"num_hidden_layers",0,&c->n_layer)||!uint_field(j,"num_attention_heads",0,&c->n_head)||!uint_field(j,"intermediate_size",0,&c->n_ff))return OC_ERR_FORMAT;
    if(!c->vocab_size||!c->n_embd||!c->n_layer||!c->n_head||!c->n_ff||c->vocab_size>10000000||c->n_embd%c->n_head)return OC_ERR_MODEL;
    if(!uint_field(j,"num_key_value_heads",c->n_head,&c->n_head_kv)||!uint_field(j,"head_dim",c->n_embd/c->n_head,&c->head_dim)||!uint_field(j,"max_position_embeddings",2048,&c->n_ctx)||!float_field(j,"rope_theta",10000,&c->rope_theta)||!float_field(j,"rms_norm_eps",1e-6f,&c->rms_norm_eps)||!flag(j,"tie_word_embeddings",false,&c->tied_embeddings))return OC_ERR_FORMAT;
    if(!c->n_head_kv||c->n_head%c->n_head_kv||!c->head_dim||c->head_dim%2||!c->n_ctx||c->head_dim>UINT32_MAX/c->n_head)return OC_ERR_MODEL;
    if(!nullish(oc_json_get(j,"rope_scaling")))return OC_ERR_MODEL;
    bool sliding=false;if(!flag(j,"use_sliding_window",false,&sliding)||sliding)return OC_ERR_MODEL;
    const OcJson *win=oc_json_get(j,"sliding_window");uint32_t w=0;
    if(!nullish(win)&&(!oc_json_u32(win,&w)||w))return OC_ERR_MODEL;
    const OcJson *act=oc_json_get(j,"hidden_act");if(act&&!eq(act,"silu"))return OC_ERR_MODEL;
    bool b=false;if(!flag(j,"mlp_bias",false,&b)||b)return OC_ERR_MODEL;
    const OcJson *partial=oc_json_get(j,"partial_rotary_factor");if(partial&&(partial->type!=OC_JSON_NUMBER||partial->number!=1))return OC_ERR_MODEL;
    c->kv_head_dim=c->head_dim;c->rope_dim=c->head_dim;c->norm_scale=1;c->expert_weights_scale=1;c->sliding_window_pattern=1;c->norm_groups=1;
    c->head_dim_swa=c->head_dim;c->n_head_kv_swa=c->n_head_kv;c->rope_dim_swa=c->rope_dim;c->rope_theta_swa=c->rope_theta;
    return OC_OK;
}
/* The existing BPE engine byte-encodes all input and has no configurable
 * prefix-space/normalization stage. Do not silently substitute a different
 * regex or SentencePiece normalization for the checkpoint's pipeline. */
static bool pipeline(const OcJson *j,bool bpe,const char **pre_name) {
    *pre_name=NULL;
    if(!nullish(oc_json_get(j,"normalizer")))return false;
    const OcJson *pre=oc_json_get(j,"pre_tokenizer"),*dec=oc_json_get(j,"decoder");
    if(bpe) {
        bool prefix,regex;
        if(eq(oc_json_get(pre,"type"),"Sequence")) {
            const OcJson *steps=oc_json_get(pre,"pretokenizers");
            if(!steps||steps->type!=OC_JSON_ARRAY||steps->count!=2)return false;
            const OcJson *split=steps->child;pre=split->next;
            if(!eq(oc_json_get(split,"type"),"Split")||!eq(oc_json_get(split,"behavior"),"Isolated")||!flag(split,"invert",false,&prefix)||prefix)return false;
            const OcJson *pattern=oc_json_get(oc_json_get(split,"pattern"),"Regex");
            const char *llama="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
            const char *qwen="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
            if(eq(pattern,llama))*pre_name="llama-bpe";else if(eq(pattern,qwen))*pre_name="qwen2";else return false;
        }
        if(!pre||!eq(oc_json_get(pre,"type"),"ByteLevel")||!flag(pre,"add_prefix_space",false,&prefix)||prefix||!flag(pre,"use_regex",true,&regex))return false;
        if(regex){if(*pre_name)return false;*pre_name="gpt2";}
        if(!dec||!eq(oc_json_get(dec,"type"),"ByteLevel"))return false;
    } else if(!nullish(pre)||!nullish(dec))return false;
    return true;
}
static bool postprocess(const OcJson *j,const OcJson *tc,const OcGgufFile *m,bool *bos) {
    const OcJson *post=oc_json_get(j,"post_processor");
    if(nullish(post)||eq(oc_json_get(post,"type"),"ByteLevel"))return true;
    if(!eq(oc_json_get(post,"type"),"TemplateProcessing"))return false;
    const OcJson *single=oc_json_get(post,"single");
    if(!single||single->type!=OC_JSON_ARRAY||single->count<1||single->count>2)return false;
    const OcJson *step=single->child;
    bool inserts_bos=single->count==2;
    if(inserts_bos) {
        const OcJson *special=oc_json_get(step,"SpecialToken");
        const OcJson *name=oc_json_get(special,"id");uint32_t tid;
        if(!name||name->type!=OC_JSON_STRING||!uint_field(special,"type_id",0,&tid)||tid)return false;
        const OcJson *definition=oc_json_get(oc_json_get(post,"special_tokens"),name->string);
        const OcJson *ids=oc_json_get(definition,"ids"),*strings=oc_json_get(definition,"tokens");uint32_t bosid,id;
        if(!ids||ids->type!=OC_JSON_ARRAY||ids->count!=1||!oc_json_u32(ids->child,&id)||!oc_gguf_metadata_get_u32(m,"tokenizer.ggml.bos_token_id",&bosid)||id!=bosid)return false;
        const OcGgufMetadataValue *tokens=oc_gguf_metadata_get(m,"tokenizer.ggml.tokens");
        if(!strings||strings->type!=OC_JSON_ARRAY||strings->count!=1||!eq(strings->child,tokens->v.arr.values[id].v.str.data))return false;
        step=step->next;
    }
    const OcJson *sequence=oc_json_get(step,"Sequence");uint32_t tid;
    if(!eq(oc_json_get(sequence,"id"),"A")||!uint_field(sequence,"type_id",0,&tid)||tid)return false;
    const OcJson *configured=oc_json_get(tc,"add_bos_token");
    if(configured&&*bos!=inserts_bos)return false;
    *bos=inserts_bos;return true;
}
static OcError tokenizer_load(const OcJson *j,const OcJson *tc,const OcJson *cfg,OcGgufFile *m,uint32_t n) {
    const OcJson *model=oc_json_get(j,"model"),*type=oc_json_get(model,"type"),*vocab=oc_json_get(model,"vocab");
    bool bpe=eq(type,"BPE");if(!bpe&&!eq(type,"Unigram"))return OC_ERR_TOKENIZER;
    const char *pre_name=NULL;
    if(!pipeline(j,bpe,&pre_name)||!vocab||vocab->type!=(bpe?OC_JSON_OBJECT:OC_JSON_ARRAY))return OC_ERR_TOKENIZER;
    bool fallback=false;if(!flag(model,"byte_fallback",false,&fallback)||fallback)return OC_ERR_TOKENIZER;
    if(bpe) {
        const OcJson *drop=oc_json_get(model,"dropout");if(!nullish(drop)&&(drop->type!=OC_JSON_NUMBER||drop->number!=0))return OC_ERR_TOKENIZER;
        const char *keys[]={"continuing_subword_prefix","end_of_word_suffix","unk_token"};
        for(size_t i=0;i<3;i++){const OcJson *v=oc_json_get(model,keys[i]);if(!nullish(v)&&!eq(v,""))return OC_ERR_TOKENIZER;}
        bool ignore=false;if(!flag(model,"ignore_merges",false,&ignore))return OC_ERR_TOKENIZER;
        OcGgufMetadataValue *iv=entry(m,"tokenizer.hf.ignore_merges",OC_GGUF_MT_BOOL);
        if(!iv)return OC_ERR_OOM;
        iv->v.b=ignore;
    }
    if(!setstr(m,"tokenizer.ggml.model",bpe?"gpt2":"llama"))return OC_ERR_OOM;
    if(pre_name&&!setstr(m,"tokenizer.ggml.pre",pre_name))return OC_ERR_OOM;
    OcGgufMetadataValue *tokens=array(m,"tokenizer.ggml.tokens",OC_GGUF_MT_STRING,n),*types=array(m,"tokenizer.ggml.token_type",OC_GGUF_MT_INT32,n),*scores=NULL;
    if(!tokens||!types)return OC_ERR_OOM;
    if(!bpe){scores=array(m,"tokenizer.ggml.scores",OC_GGUF_MT_FLOAT32,n);if(!scores)return OC_ERR_OOM;}
    size_t seq=0;
    for(const OcJson *v=vocab->child;v;v=v->next,seq++) {
        uint32_t id;const char *text;
        if(bpe){if(!oc_json_u32(v,&id))return OC_ERR_FORMAT;text=v->key;}
        else {if(seq>UINT32_MAX||v->type!=OC_JSON_ARRAY||v->count!=2||v->child->type!=OC_JSON_STRING||v->child->next->type!=OC_JSON_NUMBER||fabs(v->child->next->number)>FLT_MAX)return OC_ERR_FORMAT;id=(uint32_t)seq;text=v->child->string;}
        if(id>=n||tokens[id].v.str.data||!*text)return OC_ERR_FORMAT;
        if(!token(m,&tokens[id],text))return OC_ERR_OOM;
        types[id].v.i32=1;if(scores)scores[id].v.f32=(float)v->child->next->number;
    }
    const OcJson *added=oc_json_get(j,"added_tokens");if(added&&added->type!=OC_JSON_ARRAY)return OC_ERR_FORMAT;
    unsigned char *seen=oc_arena_alloc_bytes(m->arena,n);if(!seen)return OC_ERR_OOM;memset(seen,0,n);
    for(const OcJson *v=added?added->child:NULL;v;v=v->next) {
        uint32_t id;const OcJson *content=oc_json_get(v,"content");bool special=false,b=false;
        if(!oc_json_u32(oc_json_get(v,"id"),&id)||id>=n||seen[id]||!content||content->type!=OC_JSON_STRING||!*content->string||!flag(v,"special",false,&special))return OC_ERR_FORMAT;
        seen[id]=1;
        const char *opts[]={"single_word","lstrip","rstrip"};for(size_t k=0;k<3;k++)if(!flag(v,opts[k],false,&b)||b)return OC_ERR_TOKENIZER;
        if(!flag(v,"normalized",false,&b))return OC_ERR_TOKENIZER;
        if(tokens[id].v.str.data&&strcmp(tokens[id].v.str.data,content->string))return OC_ERR_FORMAT;
        if(!token(m,&tokens[id],content->string))return OC_ERR_OOM;
        types[id].v.i32=special?3:4;
    }
    for(uint32_t i=0;i<n;i++)if(!tokens[i].v.str.data) {
        if(!bpe)return OC_ERR_FORMAT;
        if(!token(m,&tokens[i],""))return OC_ERR_OOM;
        types[i].v.i32=5;
    }
    if(bpe) {
        const OcJson *merges=oc_json_get(model,"merges");if(!merges||merges->type!=OC_JSON_ARRAY)return OC_ERR_FORMAT;
        OcGgufMetadataValue *out=array(m,"tokenizer.ggml.merges",OC_GGUF_MT_STRING,merges->count);if(!out)return OC_ERR_OOM;
        size_t i=0;for(const OcJson *v=merges->child;v;v=v->next,i++) {
            const char *text;
            if(v->type==OC_JSON_STRING)text=v->string;
            else if(v->type==OC_JSON_ARRAY&&v->count==2&&v->child->type==OC_JSON_STRING&&v->child->next->type==OC_JSON_STRING) {
                text=oc_arena_printf(m->arena,"%s %s",v->child->string,v->child->next->string);if(!text)return OC_ERR_OOM;
            } else return OC_ERR_FORMAT;
            const char *space=strchr(text,' ');if(!space||space==text||!space[1]||strchr(space+1,' '))return OC_ERR_FORMAT;
            if(!token(m,&out[i],text))return OC_ERR_OOM;
        }
    }
    const char *names[]={"bos","eos","pad","unk","sep","cls","mask"};
    const char *ggnames[]={"bos","eos","padding","unknown","separator","cls","mask"};
    for(size_t k=0;k<7;k++) {
        char key[96];snprintf(key,sizeof(key),"%s_token_id",names[k]);const OcJson *idv=oc_json_get(cfg,key);uint32_t id=0;bool have=false;
        if(!nullish(idv)){if(!oc_json_u32(idv,&id)||id>=n)return OC_ERR_FORMAT;have=true;}
        snprintf(key,sizeof(key),"%s_token",names[k]);const OcJson *text=oc_json_get(tc,key);
        if(text&&text->type==OC_JSON_OBJECT)text=oc_json_get(text,"content");
        if(!nullish(text)) {if(text->type!=OC_JSON_STRING)return OC_ERR_FORMAT;bool found=false;for(uint32_t i=0;i<n;i++)if(!strcmp(tokens[i].v.str.data,text->string)){if(have&&id!=i)return OC_ERR_FORMAT;id=i;have=found=true;break;}if(!found)return OC_ERR_FORMAT;}
        if(k==3&&!bpe) {const OcJson *unk=oc_json_get(model,"unk_id");uint32_t uid;if(!nullish(unk)){if(!oc_json_u32(unk,&uid)||uid>=n||(have&&uid!=id))return OC_ERR_FORMAT;id=uid;have=true;}}
        if(have){if(types[id].v.i32==5)return OC_ERR_FORMAT;snprintf(key,sizeof(key),"tokenizer.ggml.%s_token_id",ggnames[k]);if(!setu32(m,key,id))return OC_ERR_OOM;types[id].v.i32=k==3?2:3;}
    }
    bool bos;if(!flag(tc,"add_bos_token",!bpe,&bos))return OC_ERR_FORMAT;
    if(!postprocess(j,tc,m,&bos))return OC_ERR_TOKENIZER;
    OcGgufMetadataValue *bv=entry(m,"tokenizer.ggml.add_bos_token",OC_GGUF_MT_BOOL);if(!bv)return OC_ERR_OOM;bv->v.b=bos;
    bool eos;if(!flag(tc,"add_eos_token",false,&eos)||eos)return OC_ERR_TOKENIZER;
    const OcJson *chat=oc_json_get(tc,"chat_template");if(!nullish(chat)){if(chat->type!=OC_JSON_STRING)return OC_ERR_TOKENIZER;if(!setstr(m,"tokenizer.chat_template",chat->string))return OC_ERR_OOM;}
    return OC_OK;
}
OcError oc_hf_metadata_load(const char *directory,OcLlamaConfig *config,OcModelArchitecture *arch,OcGgufFile *metadata) {
    if(!directory||!config||!arch||!metadata)return OC_ERR_INVALID_ARG;
    memset(config,0,sizeof(*config));memset(metadata,0,sizeof(*metadata));*arch=OC_ARCH_UNKNOWN;
    OcArena *scratch=oc_arena_new(0);if(!scratch)return OC_ERR_OOM;
    OcJson *cfg=NULL,*tok=NULL,*tc=NULL;OcError e=read_json(directory,"config.json",false,scratch,&cfg);
    if(e==OC_OK) {e=config_load(cfg,config,arch);if(e!=OC_OK)oc_log_error("HF config.json: invalid dense llama/mistral/qwen2 or qwen4exp config, or unsupported architecture, rope scaling, activation, or sliding window");}
    if(e==OC_OK)e=read_json(directory,"tokenizer.json",false,scratch,&tok);
    if(e==OC_OK)e=read_json(directory,"tokenizer_config.json",true,scratch,&tc);
    if(e==OC_OK) {metadata->arena=oc_arena_new(0);if(!metadata->arena)e=OC_ERR_OOM;}
    if(e==OC_OK) {metadata->metadata=oc_arena_alloc_bytes(metadata->arena,32*sizeof(*metadata->metadata));if(!metadata->metadata)e=OC_ERR_OOM;else memset(metadata->metadata,0,32*sizeof(*metadata->metadata));}
    if(e==OC_OK) {e=tokenizer_load(tok,tc,cfg,metadata,config->vocab_size);if(e!=OC_OK)oc_log_error("HF tokenizer.json: invalid vocabulary or unsupported normalizer, pre-tokenizer, decoder, post-processor, or added-token behavior");}
    oc_arena_free(scratch);
    if(e!=OC_OK){oc_gguf_free(metadata);memset(config,0,sizeof(*config));*arch=OC_ARCH_UNKNOWN;}
    return e;
}
