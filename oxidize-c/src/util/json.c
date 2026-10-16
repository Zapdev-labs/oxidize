#include "oxidize/json.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *end; OcArena *arena; OcError error; } Parser;
static void ws(Parser *p) { while (p->p < p->end && (*p->p == ' ' || *p->p == '\n' || *p->p == '\r' || *p->p == '\t')) p->p++; }
static void *alloc(Parser *p, size_t n) { void *v = oc_arena_alloc_bytes(p->arena, n); if (!v) p->error = OC_ERR_OOM; else memset(v, 0, n); return v; }
static bool hex4(Parser *p, unsigned *out) {
    unsigned v = 0;
    for (int i=0;i<4;i++) { if(p->p==p->end) return false; unsigned c=(unsigned char)*p->p++; unsigned d;
        if(c>='0'&&c<='9') d=c-'0'; else if(c>='a'&&c<='f') d=c-'a'+10; else if(c>='A'&&c<='F') d=c-'A'+10; else return false; v=v*16+d; }
    *out=v; return true;
}
static char *string(Parser *p) {
    if(p->p==p->end || *p->p++!='"') return NULL;
    const char *start=p->p, *q=start; bool escaped=false;
    while(q<p->end) { char c=*q++; if(!escaped&&c=='"') break; if(!escaped&&c=='\\') escaped=true; else escaped=false; }
    if(q==start || q[-1]!='"') return NULL;
    char *s=alloc(p,(size_t)(q-start)+1); if(!s) return NULL; size_t n=0;
    while(p->p<p->end) {
        unsigned c=(unsigned char)*p->p++;
        if(c=='"') { s[n]=0; return s; }
        if(c<32) return NULL;
        if(c=='\\') {
            if(p->p==p->end) return NULL;
            c=(unsigned char)*p->p++;
            if(c=='u') { if(!hex4(p,&c)) return NULL;
                if(c>=0xd800&&c<=0xdbff) { unsigned lo; if(p->end-p->p<2||p->p[0]!='\\'||p->p[1]!='u') return NULL; p->p+=2; if(!hex4(p,&lo)||lo<0xdc00||lo>0xdfff) return NULL; c=0x10000+((c-0xd800)<<10)+(lo-0xdc00); }
                else if(c>=0xdc00&&c<=0xdfff) return NULL;
                if(!c) return NULL;
                if(c<128) s[n++]=(char)c;
                else { if(c>=0x10000) { s[n++]=(char)(0xf0|(c>>18)); s[n++]=(char)(0x80|((c>>12)&63)); } else if(c>=0x800) s[n++]=(char)(0xe0|(c>>12));
                    else { s[n++]=(char)(0xc0|(c>>6)); s[n++]=(char)(0x80|(c&63)); continue; }
                    s[n++]=(char)(0x80|((c>>6)&63)); s[n++]=(char)(0x80|(c&63)); }
                continue;
            }
            switch(c) { case '"': case '\\': case '/': break; case 'b':c=8;break;case 'f':c=12;break;case 'n':c=10;break;case 'r':c=13;break;case 't':c=9;break;default:return NULL; }
        } else if(c>=128) {
            unsigned cp; int extra;
            if(c>=0xc2&&c<=0xdf) { cp=c&31; extra=1; } else if(c>=0xe0&&c<=0xef) { cp=c&15;extra=2; } else if(c>=0xf0&&c<=0xf4) {cp=c&7;extra=3;} else return NULL;
            s[n++]=(char)c;
            for(int i=0;i<extra;i++) { if(p->p==p->end) return NULL; unsigned b=(unsigned char)*p->p++; if((b&0xc0)!=0x80) return NULL; cp=(cp<<6)|(b&63);s[n++]=(char)b; }
            if((extra==2&&cp<0x800)||(extra==3&&cp<0x10000)||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)) return NULL;
            continue;
        }
        s[n++]=(char)c;
    }
    return NULL;
}
/* OC_OK when keys are unique; OC_ERR_FORMAT on a duplicate; OC_ERR_OOM when
 * the scratch table cannot be allocated. The probe bound stops a crafted
 * document full of FNV-1a-colliding keys from stalling the parse: once a
 * single key needs more than MAX_PROBE steps the input is rejected. */
static OcError unique(OcJson *v) {
    size_t cap=8; while(cap<v->count*2) cap*=2;
    const char **keys=calloc(cap,sizeof(*keys)); if(!keys) return OC_ERR_OOM;
    OcError err=OC_OK;
    for(OcJson *c=v->child;c&&err==OC_OK;c=c->next) {
        size_t h=2166136261u;for(const unsigned char *s=(const unsigned char *)c->key;*s;s++) h=(h^*s)*16777619u;h&=cap-1;
        size_t probes=0;
        while(keys[h]) {
            if(!strcmp(keys[h],c->key)||++probes>64) {err=OC_ERR_FORMAT;break;}
            h=(h+1)&(cap-1);
        }
        if(err==OC_OK) keys[h]=c->key;
    }
    free(keys);return err;
}
static OcJson *value(Parser *p,unsigned depth) {
    ws(p);if(depth>128||p->p==p->end) return NULL;
    OcJson *v=alloc(p,sizeof(*v));if(!v)return NULL;
    char c=*p->p;
    if(c=='"') { v->type=OC_JSON_STRING;v->string=string(p);return v->string?v:NULL; }
    if(c=='{'||c=='[') { p->p++;v->type=c=='{'?OC_JSON_OBJECT:OC_JSON_ARRAY;char close=c=='{'?'}':']';OcJson **tail=&v->child;ws(p);
        if(p->p<p->end&&*p->p==close) {p->p++;return v;}
        for(;;) { char *key=NULL; if(c=='{') { ws(p);key=string(p);if(!key)return NULL;ws(p);if(p->p==p->end||*p->p++!=':')return NULL; }
            OcJson *child=value(p,depth+1);if(!child)return NULL;child->key=key;*tail=child;tail=&child->next;v->count++;ws(p);
            if(p->p==p->end)return NULL;
            if(*p->p==close) {p->p++;break;} if(*p->p++!=',')return NULL;
        }
        if(c=='{'){OcError ue=unique(v);if(ue!=OC_OK){p->error=ue;return NULL;}} return v;
    }
    const char *literal=c=='t'?"true":c=='f'?"false":c=='n'?"null":NULL;
    if(literal) { size_t n=strlen(literal);if((size_t)(p->end-p->p)<n||memcmp(p->p,literal,n))return NULL;p->p+=n;v->type=c=='n'?OC_JSON_NULL:OC_JSON_BOOL;v->boolean=c=='t';return v; }
    const char *start=p->p;if(*p->p=='-')p->p++;if(p->p==p->end)return NULL;
    if(*p->p=='0')p->p++;else {if(*p->p<'1'||*p->p>'9')return NULL;while(p->p<p->end&&*p->p>='0'&&*p->p<='9')p->p++;}
    if(p->p<p->end&&*p->p=='.') {p->p++;const char *a=p->p;while(p->p<p->end&&*p->p>='0'&&*p->p<='9')p->p++;if(a==p->p)return NULL;}
    if(p->p<p->end&&(*p->p=='e'||*p->p=='E')) {p->p++;if(p->p<p->end&&(*p->p=='+'||*p->p=='-'))p->p++;const char *a=p->p;while(p->p<p->end&&*p->p>='0'&&*p->p<='9')p->p++;if(a==p->p)return NULL;}
    size_t n=(size_t)(p->p-start);if(n>127)return NULL;char buf[128];memcpy(buf,start,n);buf[n]=0;errno=0;char *end;v->number=strtod(buf,&end);/* ERANGE on underflow to a denormal (e.g. 1e-310) is still a valid, representable number — accept it; but reject non-finite overflow and literals that underflow all the way to zero (1e-999). */if(*end||!isfinite(v->number)||(errno==ERANGE&&v->number==0.0))return NULL;v->type=OC_JSON_NUMBER;return v;
}
OcError oc_json_parse(const char *data,size_t len,OcArena *arena,OcJson **out) {
    if(!data||!arena||!out)return OC_ERR_INVALID_ARG;
    *out=NULL;Parser p={data,data+len,arena,OC_ERR_FORMAT};OcJson *v=value(&p,0);ws(&p);if(!v||p.p!=p.end)return p.error;*out=v;return OC_OK;
}
const OcJson *oc_json_get(const OcJson *o,const char *key) {if(!o||o->type!=OC_JSON_OBJECT)return NULL;for(const OcJson *v=o->child;v;v=v->next)if(!strcmp(v->key,key))return v;return NULL;}
bool oc_json_u32(const OcJson *v,uint32_t *out) {if(!v||v->type!=OC_JSON_NUMBER||v->number<0||v->number>UINT32_MAX||floor(v->number)!=v->number)return false;*out=(uint32_t)v->number;return true;}
