/* test_hashtable.c — OcHashtable (FNV-1a, open-addressing) tests. */
#include <criterion/criterion.h>
#include "oxidize/hashtable.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

Test(hashtable, fnv1a_known_values)
{
    /* FNV-1a 64-bit: hash of "" is the offset basis; "a" multiplies once. */
    cr_assert_eq(oc_fnv1a_hash(""), 0xcbf29ce484222325ULL, "empty hash");
    /* hash("a") = (basis ^ 'a') * prime */
    uint64_t expected = (0xcbf29ce484222325ULL ^ (uint64_t)'a') * 0x100000001b3ULL;
    cr_assert_eq(oc_fnv1a_hash("a"), expected, "single-char hash");
}

Test(hashtable, put_get_basic)
{
    OcHashtable *ht = oc_hashtable_new(0);
    cr_assert_not_null(ht, "");
    int v1 = 10, v2 = 20, v3 = 30;
    cr_assert_eq(oc_hashtable_put(ht, "one",   &v1, NULL), OC_OK, "");
    cr_assert_eq(oc_hashtable_put(ht, "two",   &v2, NULL), OC_OK, "");
    cr_assert_eq(oc_hashtable_put(ht, "three", &v3, NULL), OC_OK, "");

    void *out = NULL;
    cr_assert(oc_hashtable_get(ht, "one",   &out), "one should exist");
    cr_assert_eq(out, &v1, "one value");
    cr_assert(oc_hashtable_get(ht, "two",   &out), "two should exist");
    cr_assert_eq(out, &v2, "two value");
    cr_assert(oc_hashtable_get(ht, "three", &out), "three should exist");
    cr_assert_eq(out, &v3, "three value");
    cr_assert(!oc_hashtable_get(ht, "missing", &out), "missing should not exist");

    cr_assert_eq(oc_hashtable_size(ht), 3, "size");
    oc_hashtable_free(ht);
}

Test(hashtable, put_replaces)
{
    OcHashtable *ht = oc_hashtable_new(0);
    int v1 = 1, v2 = 2;
    void *prev = NULL;
    cr_assert_eq(oc_hashtable_put(ht, "k", &v1, NULL), OC_OK, "");
    cr_assert_eq(oc_hashtable_put(ht, "k", &v2, &prev), OC_OK, "");
    cr_assert_eq(prev, &v1, "prev should be v1");
    void *out = NULL;
    cr_assert(oc_hashtable_get(ht, "k", &out), "");
    cr_assert_eq(out, &v2, "should be v2 now");
    cr_assert_eq(oc_hashtable_size(ht), 1, "size still 1");
    oc_hashtable_free(ht);
}

Test(hashtable, remove)
{
    OcHashtable *ht = oc_hashtable_new(0);
    int v = 7;
    oc_hashtable_put(ht, "k", &v, NULL);
    void *prev = NULL;
    cr_assert(oc_hashtable_remove(ht, "k", &prev), "remove should succeed");
    cr_assert_eq(prev, &v, "");
    cr_assert(!oc_hashtable_get(ht, "k", NULL), "should be gone");
    cr_assert_eq(oc_hashtable_size(ht), 0, "");
    cr_assert(!oc_hashtable_remove(ht, "k", NULL), "remove again should fail");
    oc_hashtable_free(ht);
}

Test(hashtable, grows_dynamically)
{
    /* Insert many entries to force multiple grows. Default cap is 16; insert
     * 1000 to verify growth + rehash preserves all entries. */
    OcHashtable *ht = oc_hashtable_new(4);
    cr_assert_not_null(ht, "");
    static int vals[1000];
    char keybuf[32];
    for (int i = 0; i < 1000; i++) {
        vals[i] = i;
        snprintf(keybuf, sizeof(keybuf), "key_%d", i);
        OcError e = oc_hashtable_put(ht, keybuf, &vals[i], NULL);
        cr_assert_eq(e, OC_OK, "put %d", i);
    }
    cr_assert_eq(oc_hashtable_size(ht), 1000, "size");
    cr_assert(oc_hashtable_capacity(ht) >= 1000, "cap grew");

    /* Verify all entries are retrievable. */
    for (int i = 0; i < 1000; i++) {
        snprintf(keybuf, sizeof(keybuf), "key_%d", i);
        void *out = NULL;
        cr_assert(oc_hashtable_get(ht, keybuf, &out), "get %d", i);
        cr_assert_eq(out, &vals[i], "val %d", i);
    }

    /* Remove half, verify the rest remain. */
    for (int i = 0; i < 500; i++) {
        snprintf(keybuf, sizeof(keybuf), "key_%d", i);
        cr_assert(oc_hashtable_remove(ht, keybuf, NULL), "rm %d", i);
    }
    cr_assert_eq(oc_hashtable_size(ht), 500, "size after rm");
    for (int i = 500; i < 1000; i++) {
        snprintf(keybuf, sizeof(keybuf), "key_%d", i);
        cr_assert(oc_hashtable_get(ht, keybuf, NULL), "get %d after rm", i);
    }
    oc_hashtable_free(ht);
}

Test(hashtable, iterator_visits_all)
{
    OcHashtable *ht = oc_hashtable_new(0);
    int v1 = 1, v2 = 2, v3 = 3;
    oc_hashtable_put(ht, "a", &v1, NULL);
    oc_hashtable_put(ht, "b", &v2, NULL);
    oc_hashtable_put(ht, "c", &v3, NULL);

    size_t iter = 0;
    const char *k = NULL;
    void *v = NULL;
    int seen = 0;
    char seen_keys[3] = {0, 0, 0};
    while (oc_hashtable_next(ht, &iter, &k, &v)) {
        cr_assert_not_null(k, "");
        cr_assert_not_null(v, "");
        for (int i = 0; i < 3; i++) {
            char want = 'a' + (char)i;
            if (k[0] == want && k[1] == '\0' && !seen_keys[i]) {
                seen_keys[i] = 1;
                seen++;
            }
        }
    }
    cr_assert_eq(seen, 3, "should have seen 3 distinct keys");
    oc_hashtable_free(ht);
}

