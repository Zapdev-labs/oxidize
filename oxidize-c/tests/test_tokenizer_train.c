/* test_tokenizer_train.c — Criterion tests for the BPE tokenizer trainer. */

#define _POSIX_C_SOURCE 200809L

#include <criterion/criterion.h>
#include <criterion/redirect.h>

#include "oxidize/tokenizer_train.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Helper: create a temp file path and return it (static buffer). */
static const char *make_temp_path(const char *suffix)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oc_bpe_train_%d_%s", (int)getpid(), suffix);
    return path;
}


Test(bpe_train, init_basic)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 500, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    cr_assert_not_null(t, "trainer should not be NULL");
    cr_assert_eq(oc_bpe_trainer_vocab_size(t), 0, "untrained vocab should be empty");
    cr_assert_eq(oc_bpe_trainer_merge_count(t), 0, "untrained merges should be empty");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, init_default_config_values)
{
    /* All-zero config should use defaults. */
    OcBpeTrainConfig cfg = { 0 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    cr_assert_not_null(t, "");
    /* Train on a minimal corpus to verify it doesn't crash. */
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    /* Vocab should contain 'a' and 'b' at minimum. */
    cr_assert_geq(oc_bpe_trainer_vocab_size(t), 2, "");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, init_null_config_trainer)
{
    /* init() takes a config by value, so we test the NULL trainer case via
     * the other functions. */
    cr_assert_eq(oc_bpe_trainer_train(NULL, "hello", 0), OC_ERR_INVALID_ARG, "");
}

Test(bpe_train, free_null_safe)
{
    oc_bpe_trainer_free(NULL);
}


Test(bpe_train, train_simple_corpus)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    /* "ab ab ab" — the pair (a, b) appears 3 times. */
    OcError e = oc_bpe_trainer_train(t, "ab ab ab", 0);
    cr_assert_eq(e, OC_OK, "train failed: %s", oc_error_msg(e));
    cr_assert_geq(oc_bpe_trainer_vocab_size(t), 2, "vocab should have at least 2 entries");
    cr_assert_geq(oc_bpe_trainer_merge_count(t), 1, "should have at least 1 merge");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, train_empty_corpus_fails)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 100, .min_frequency = 2,
                             .max_merges = 10 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    OcError e = oc_bpe_trainer_train(t, "", 0);
    cr_assert_neq(e, OC_OK, "empty corpus should fail");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, train_null_corpus_fails)
{
    OcBpeTrainer *t = oc_bpe_trainer_init((OcBpeTrainConfig){ 0 });
    OcError e = oc_bpe_trainer_train(t, NULL, 0);
    cr_assert_neq(e, OC_OK, "NULL corpus should fail");
    oc_bpe_trainer_free(t);
}


Test(bpe_train, vocab_contains_individual_chars)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "abc abc abc", 0);

    const OcBpeVocabEntry *entries;
    size_t count;
    OcError e = oc_bpe_trainer_vocab(t, &entries, &count);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_gt(count, 0, "");

    /* Check that 'a', 'b', 'c' are in the vocab. */
    bool found_a = false, found_b = false, found_c = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(entries[i].token, "a") == 0) found_a = true;
        if (strcmp(entries[i].token, "b") == 0) found_b = true;
        if (strcmp(entries[i].token, "c") == 0) found_c = true;
    }
    cr_assert(found_a, "vocab should contain 'a'");
    cr_assert(found_b, "vocab should contain 'b'");
    cr_assert(found_c, "vocab should contain 'c'");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, vocab_ids_are_sequential)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "ab ab ab", 0);

    const OcBpeVocabEntry *entries;
    size_t count;
    oc_bpe_trainer_vocab(t, &entries, &count);
    for (size_t i = 0; i < count; i++) {
        cr_assert_eq(entries[i].id, (uint32_t)i, "vocab id should match index");
    }
    oc_bpe_trainer_free(t);
}

Test(bpe_train, vocab_getter_null_args)
{
    cr_assert_eq(oc_bpe_trainer_vocab(NULL, NULL, NULL), OC_ERR_INVALID_ARG, "");
}


Test(bpe_train, merge_rule_correctness)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    /* "ab ab ab" — first merge should be (a, b) -> "ab". */
    oc_bpe_trainer_train(t, "ab ab ab", 0);

    const OcBpeMerge *merges;
    size_t count;
    OcError e = oc_bpe_trainer_merges(t, &merges, &count);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_geq(count, 1, "should have at least 1 merge");

    /* The first merge should be (a, b) -> "ab". */
    /* Find the vocab ids for 'a' and 'b'. */
    const OcBpeVocabEntry *entries;
    size_t vcount;
    oc_bpe_trainer_vocab(t, &entries, &vcount);

    uint32_t id_a = 0, id_b = 0, id_ab = 0;
    for (size_t i = 0; i < vcount; i++) {
        if (strcmp(entries[i].token, "a") == 0) id_a = entries[i].id;
        if (strcmp(entries[i].token, "b") == 0) id_b = entries[i].id;
        if (strcmp(entries[i].token, "ab") == 0) id_ab = entries[i].id;
    }

    cr_assert_eq(merges[0].left, id_a, "first merge left should be 'a'");
    cr_assert_eq(merges[0].right, id_b, "first merge right should be 'b'");
    cr_assert_eq(merges[0].merged, id_ab, "first merge result should be 'ab'");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, merges_getter_null_args)
{
    cr_assert_eq(oc_bpe_trainer_merges(NULL, NULL, NULL), OC_ERR_INVALID_ARG, "");
}

Test(bpe_train, multiple_merges)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    /* "abcabcabc abcabcabc" — should produce multiple merges: (a,b), then
     * (ab, c), etc. */
    oc_bpe_trainer_train(t, "abcabcabc abcabcabc", 0);
    cr_assert_geq(oc_bpe_trainer_merge_count(t), 2, "should have at least 2 merges");
    oc_bpe_trainer_free(t);
}


Test(bpe_train, max_merges_limit)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 3 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    /* A long corpus with many possible merges. */
    oc_bpe_trainer_train(t, "abcdefgh abcdefgh abcdefgh abcdefgh", 0);
    cr_assert_leq(oc_bpe_trainer_merge_count(t), 3, "should not exceed max_merges");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, max_vocab_size_limit)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 10, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "abcdefgh abcdefgh abcdefgh", 0);
    cr_assert_leq(oc_bpe_trainer_vocab_size(t), 10, "should not exceed max_vocab_size");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, min_frequency_threshold)
{
    /* With min_frequency=10, no merges should happen (each pair appears < 10). */
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 10,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    cr_assert_eq(oc_bpe_trainer_merge_count(t), 0, "no merges when min_freq too high");
    oc_bpe_trainer_free(t);
}


Test(bpe_train, save_creates_file)
{
    const char *path = make_temp_path("vocab.json");
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    OcError e = oc_bpe_trainer_save(t, path);
    cr_assert_eq(e, OC_OK, "save failed: %s", oc_error_msg(e));

    /* File should exist and be non-empty. */
    FILE *fp = fopen(path, "rb");
    cr_assert_not_null(fp, "file should exist");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);
    cr_assert_gt(size, 0, "file should be non-empty");
    unlink(path);
    oc_bpe_trainer_free(t);
}

Test(bpe_train, save_null_args)
{
    OcBpeTrainer *t = oc_bpe_trainer_init((OcBpeTrainConfig){ 0 });
    cr_assert_neq(oc_bpe_trainer_save(t, NULL), OC_OK, "");
    cr_assert_neq(oc_bpe_trainer_save(NULL, "foo.json"), OC_OK, "");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, save_json_contains_vocab_and_merges)
{
    const char *path = make_temp_path("check.json");
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    oc_bpe_trainer_save(t, path);

    /* Read the file and check it contains "vocab" and "merges" keys. */
    FILE *fp = fopen(path, "rb");
    cr_assert_not_null(fp, "");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    cr_assert_not_null(buf, "");
    size_t nread = fread(buf, 1, size, fp);
    buf[nread] = '\0';
    fclose(fp);

    cr_assert(strstr(buf, "\"vocab\"") != NULL, "JSON should contain 'vocab' key");
    cr_assert(strstr(buf, "\"merges\"") != NULL, "JSON should contain 'merges' key");
    cr_assert(strstr(buf, "\"token\"") != NULL, "JSON should contain 'token' key");
    cr_assert(strstr(buf, "\"id\"") != NULL, "JSON should contain 'id' key");
    cr_assert(strstr(buf, "\"left\"") != NULL, "JSON should contain 'left' key");
    cr_assert(strstr(buf, "\"right\"") != NULL, "JSON should contain 'right' key");

    free(buf);
    unlink(path);
    oc_bpe_trainer_free(t);
}


Test(bpe_train, punctuation_handled)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "hello, world! hello, world!", 0);
    /* Vocab should contain punctuation chars. */
    const OcBpeVocabEntry *entries;
    size_t count;
    oc_bpe_trainer_vocab(t, &entries, &count);
    bool found_comma = false, found_excl = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(entries[i].token, ",") == 0) found_comma = true;
        if (strcmp(entries[i].token, "!") == 0) found_excl = true;
    }
    cr_assert(found_comma, "vocab should contain ','");
    cr_assert(found_excl, "vocab should contain '!'");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, repeated_chars_merge)
{
    /* "aaaa aaaa" — the pair (a, a) should be the most frequent. */
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "aaaa aaaa", 0);
    cr_assert_geq(oc_bpe_trainer_merge_count(t), 1, "should merge (a,a)");
    /* Check that "aa" is in the vocab. */
    const OcBpeVocabEntry *entries;
    size_t count;
    oc_bpe_trainer_vocab(t, &entries, &count);
    bool found_aa = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(entries[i].token, "aa") == 0) found_aa = true;
    }
    cr_assert(found_aa, "vocab should contain 'aa'");
    oc_bpe_trainer_free(t);
}


Test(bpe_train, retrain_resets_state)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "ab ab ab", 0);

    /* Retrain with a different corpus (same structure: 2 unique chars). */
    oc_bpe_trainer_train(t, "xy xy xy", 0);

    /* The vocab should be different (xy chars, not ab). */
    const OcBpeVocabEntry *entries;
    size_t count;
    oc_bpe_trainer_vocab(t, &entries, &count);
    bool found_x = false, found_a = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(entries[i].token, "x") == 0) found_x = true;
        if (strcmp(entries[i].token, "a") == 0) found_a = true;
    }
    cr_assert(found_x, "retrained vocab should contain 'x'");
    cr_assert(!found_a, "retrained vocab should NOT contain 'a'");

    /* Vocab should contain 'xy' (the merge of x+y, the most frequent pair). */
    bool found_xy = false;
    for (size_t i = 0; i < count; i++) {
        if (strcmp(entries[i].token, "xy") == 0) found_xy = true;
    }
    cr_assert(found_xy, "retrained vocab should contain 'xy' merge");

    oc_bpe_trainer_free(t);
}


Test(bpe_train, multi_word_corpus)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "the cat sat on the mat the cat sat", 0);
    cr_assert_geq(oc_bpe_trainer_vocab_size(t), 8, "should have at least 8 chars");
    /* "th" and "he" are common pairs in "the" x3, so we expect at least one merge. */
    cr_assert_geq(oc_bpe_trainer_merge_count(t), 1, "should have merges for 'the'");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, single_char_no_merges)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 1,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    oc_bpe_trainer_train(t, "a a a a a", 0);
    /* No adjacent pairs within a word, so no merges. */
    cr_assert_eq(oc_bpe_trainer_merge_count(t), 0, "single-char words have no pairs");
    cr_assert_eq(oc_bpe_trainer_vocab_size(t), 1, "vocab should contain only 'a'");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, vocab_size_function)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    cr_assert_eq(oc_bpe_trainer_vocab_size(t), 0, "untrained should be 0");
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    cr_assert_geq(oc_bpe_trainer_vocab_size(t), 2, "");
    cr_assert_eq(oc_bpe_trainer_vocab_size(NULL), 0, "NULL should return 0");
    oc_bpe_trainer_free(t);
}

Test(bpe_train, merge_count_function)
{
    OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
                             .max_merges = 100 };
    OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
    cr_assert_eq(oc_bpe_trainer_merge_count(t), 0, "untrained should be 0");
    oc_bpe_trainer_train(t, "ab ab ab", 0);
    cr_assert_geq(oc_bpe_trainer_merge_count(t), 1, "");
    cr_assert_eq(oc_bpe_trainer_merge_count(NULL), 0, "NULL should return 0");
    oc_bpe_trainer_free(t);
}
