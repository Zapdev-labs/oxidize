#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/hf_metadata.h"
#include "oxidize/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static const char *config="{\"model_type\":\"llama\",\"vocab_size\":5,\"hidden_size\":8,\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"intermediate_size\":16,\"max_position_embeddings\":128,\"tie_word_embeddings\":true,\"bos_token_id\":3,\"eos_token_id\":4}";
static const char *bpe="{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},\"merges\":[[\"a\",\"b\"]]},\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false},\"decoder\":{\"type\":\"ByteLevel\"},\"added_tokens\":[{\"id\":3,\"content\":\"<s>\",\"special\":true},{\"id\":4,\"content\":\"</s>\",\"special\":true}]}";
static void write_file(const char *dir,const char *name,const char *s) {char path[256];snprintf(path,sizeof(path),"%s/%s",dir,name);FILE *f=fopen(path,"wb");cr_assert_not_null(f);cr_assert_eq(fwrite(s,1,strlen(s),f),strlen(s));cr_assert_eq(fclose(f),0);}
static void cleanup(const char *dir) {const char *names[]={"config.json","tokenizer.json","tokenizer_config.json"};for(size_t i=0;i<3;i++){char path[256];snprintf(path,sizeof(path),"%s/%s",dir,names[i]);unlink(path);}rmdir(dir);}

Test(hf_metadata, llama3_split_and_ignore_merges)
{
    const char *inputs[] = {"{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},\"merges\":[],\"ignore_merges\":false},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]},\"decoder\":{\"type\":\"ByteLevel\"},\"added_tokens\":[{\"id\":3,\"content\":\"<s>\",\"special\":true},{\"id\":4,\"content\":\"</s>\",\"special\":true}]}", "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},\"merges\":[],\"ignore_merges\":true},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]},\"decoder\":{\"type\":\"ByteLevel\"},\"added_tokens\":[{\"id\":3,\"content\":\"<s>\",\"special\":true},{\"id\":4,\"content\":\"</s>\",\"special\":true}]}"};
    char dir[] = "/tmp/oc-hf-XXXXXX";
    cr_assert_not_null(mkdtemp(dir));
    write_file(dir, "config.json", config);
    for (size_t mode=0; mode<2; mode++) {
        write_file(dir, "tokenizer.json", inputs[mode]);
        OcLlamaConfig c; OcModelArchitecture arch; OcGgufFile m={0};
        cr_assert_eq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);
        const char *pre=NULL;
        cr_assert(oc_gguf_metadata_get_str(&m,"tokenizer.ggml.pre",&pre,NULL));
        cr_assert_str_eq(pre,"llama-bpe");
        OcTokenizer t={0};
        cr_assert_eq(oc_tokenizer_load_from_gguf(&m,&t),OC_OK);
        uint32_t *ids=NULL; size_t n=0;
        cr_assert_eq(oc_tokenizer_encode(&t,"ab",OC_TOK_DISALLOW_SPECIAL,&ids,&n),OC_OK);
        cr_assert_eq(n,mode ? 1 : 2);
        cr_assert_eq(ids[0],mode ? 2 : 0);
        if(!mode) cr_assert_eq(ids[1],1);
        free(ids); oc_tokenizer_free(&t); oc_gguf_free(&m);
    }
    cleanup(dir);
}
static void roundtrip(OcGgufFile *m) {OcTokenizer t={0};cr_assert_eq(oc_tokenizer_load_from_gguf(m,&t),OC_OK);uint32_t *ids=NULL;size_t n=0;char *text=NULL;cr_assert_eq(oc_tokenizer_encode(&t,"ab",OC_TOK_DISALLOW_SPECIAL,&ids,&n),OC_OK);cr_assert_eq(n,1);cr_assert_eq(ids[0],2);cr_assert_eq(oc_tokenizer_decode(&t,ids,n,&text),OC_OK);cr_assert_str_eq(text,"ab");free(ids);free(text);oc_tokenizer_free(&t);}
Test(hf_metadata, bpe_native_roundtrip_and_added_tokens) {
    char dir[]="/tmp/oc-hf-XXXXXX";cr_assert_not_null(mkdtemp(dir));write_file(dir,"config.json",config);write_file(dir,"tokenizer.json",bpe);write_file(dir,"tokenizer_config.json","{\"add_bos_token\":true,\"bos_token\":\"<s>\",\"eos_token\":{\"content\":\"</s>\"},\"chat_template\":\"{{ messages }}\"}");
    OcLlamaConfig c={0};OcModelArchitecture arch;OcGgufFile m={0};cr_assert_eq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);cr_assert_eq(arch,OC_ARCH_LLAMA);cr_assert_eq(c.n_head_kv,1);cr_assert_eq(c.head_dim,4);cr_assert(c.tied_embeddings);cr_assert_eq(m.tensor_count,0);cr_assert_null(m.backing_buf);cr_assert_not_null(oc_gguf_metadata_get(&m,"tokenizer.chat_template"));roundtrip(&m);
    OcTokenizer t={0};cr_assert_eq(oc_tokenizer_load_from_gguf(&m,&t),OC_OK);uint32_t *ids=NULL;size_t n=0;cr_assert_eq(oc_tokenizer_encode(&t,"<s>",OC_TOK_ALLOW_SPECIAL,&ids,&n),OC_OK);cr_assert_eq(n,1);cr_assert_eq(ids[0],3);free(ids);oc_tokenizer_free(&t);oc_gguf_free(&m);cleanup(dir);
}
Test(hf_metadata, raw_unigram_roundtrip) {
    char dir[]="/tmp/oc-hf-XXXXXX";cr_assert_not_null(mkdtemp(dir));write_file(dir,"config.json",config);write_file(dir,"tokenizer.json","{\"model\":{\"type\":\"Unigram\",\"unk_id\":3,\"vocab\":[[\"a\",-3],[\"b\",-3],[\"ab\",-1],[\"<s>\",0],[\"</s>\",0]]}}");
    OcLlamaConfig c;OcModelArchitecture arch;OcGgufFile m={0};cr_assert_eq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);roundtrip(&m);oc_gguf_free(&m);cleanup(dir);
}

Test(hf_metadata, padded_weight_vocabulary_does_not_invent_token_text)
{
    char dir[]="/tmp/oc-hf-XXXXXX";
    cr_assert_not_null(mkdtemp(dir));
    write_file(dir,"config.json","{\"model_type\":\"llama\",\"vocab_size\":6,\"hidden_size\":8,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"intermediate_size\":16}");
    write_file(dir,"tokenizer.json",bpe);
    OcLlamaConfig c; OcModelArchitecture arch; OcGgufFile m={0};
    cr_assert_eq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);
    OcTokenizer t={0};
    cr_assert_eq(oc_tokenizer_load_from_gguf(&m,&t),OC_OK);
    uint32_t id=5; char *text=NULL;
    cr_assert_eq(oc_tokenizer_decode(&t,&id,1,&text),OC_OK);
    cr_assert_str_eq(text,"");
    free(text); oc_tokenizer_free(&t); oc_gguf_free(&m); cleanup(dir);
}
Test(hf_metadata, rejects_bad_json_config_and_semantics) {
    const char *bad[]={"{\"model_type\":\"llama\",\"model_type\":\"llama\"}","{\"model_type\":\"gemma\"}","{\"model_type\":\"llama\",\"hidden_size\":-1}","{}"};
    char dir[]="/tmp/oc-hf-XXXXXX";cr_assert_not_null(mkdtemp(dir));write_file(dir,"tokenizer.json",bpe);
    for(size_t i=0;i<4;i++){write_file(dir,"config.json",bad[i]);OcLlamaConfig c;OcModelArchitecture arch;OcGgufFile m={0};cr_assert_neq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);cr_assert_null(m.arena);cr_assert_eq(c.n_embd,0);cr_assert_eq(arch,OC_ARCH_UNKNOWN);}
    write_file(dir,"config.json",config);
    const char *tokens[]={"{\"model\":{\"type\":\"WordPiece\"}}","{\"normalizer\":{\"type\":\"NFC\"},\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}","{\"model\":{\"type\":\"Unigram\",\"vocab\":[[\"a\",1]]},\"pre_tokenizer\":{\"type\":\"Metaspace\"}}","{"};
    for(size_t i=0;i<4;i++){write_file(dir,"tokenizer.json",tokens[i]);OcLlamaConfig c;OcModelArchitecture arch;OcGgufFile m={0};cr_assert_neq(oc_hf_metadata_load(dir,&c,&arch,&m),OC_OK);cr_assert_null(m.arena);cr_assert_eq(c.n_embd,0);}
    cleanup(dir);
}
