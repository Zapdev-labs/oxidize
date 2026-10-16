#include <criterion/criterion.h>
#include "oxidize/json.h"
#include <string.h>
Test(json, unicode_and_numbers) {
    OcArena *a=oc_arena_new(0);OcJson *j=NULL;
    const char *s="{\"x\":\"\\uD83D\\uDE00\\n\",\"n\":4294967295,\"a\":[true,null,-1.25e2]}";
    cr_assert_eq(oc_json_parse(s,strlen(s),a,&j),OC_OK);
    cr_assert_str_eq(oc_json_get(j,"x")->string,"\xf0\x9f\x98\x80\n");
    uint32_t n;cr_assert(oc_json_u32(oc_json_get(j,"n"),&n));cr_assert_eq(n,UINT32_MAX);oc_arena_free(a);
}
Test(json, rejects_malformed_and_duplicate_keys) {
    const char *bad[]={"{\"x\":1,\"\\u0078\":2}","[1,]","{\"x\":01}","1e999","1e-999","\"\\uD800\"","\"\\uDC00\"","\"\\u0000\"","\"\xc0\x80\"","true false","{\"x\":1","\"unterminated","-","1.","[1:2]"};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++){OcArena *a=oc_arena_new(0);OcJson *j=NULL;cr_assert_neq(oc_json_parse(bad[i],strlen(bad[i]),a,&j),OC_OK,"%s",bad[i]);cr_assert_null(j);oc_arena_free(a);}
}
Test(json, bounded_integer) {
    const char *bad[]={"4294967296","-1","1.5"};
    for(size_t i=0;i<3;i++){OcArena *a=oc_arena_new(0);OcJson *j=NULL;uint32_t n;cr_assert_eq(oc_json_parse(bad[i],strlen(bad[i]),a,&j),OC_OK);cr_assert_not(oc_json_u32(j,&n));oc_arena_free(a);}
}
Test(json, depth_limit) {
    char s[600];memset(s,'[',150);s[150]='0';memset(s+151,']',150);s[301]=0;
    OcArena *a=oc_arena_new(0);OcJson *j=NULL;cr_assert_eq(oc_json_parse(s,301,a,&j),OC_ERR_FORMAT);oc_arena_free(a);
}
