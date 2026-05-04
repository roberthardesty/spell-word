/**
 * @file test_decoder_core.c
 * @brief Unit tests for the host-portable word decoder.
 *
 * Synthesises top-K events with controlled probabilities and a tiny
 * dictionary, then asserts that dec_resolve_full_eow() picks the right
 * word (or abstains, depending on the case). No real audio, no IDF.
 *
 * Build + run:
 *   cd tools/decoder_replay && make test
 *
 * Each test prints "PASS: <name>" or aborts with a detailed FAIL message.
 * Exit code 0 = all pass.
 *
 * The seven acceptance cases listed in Vikunja #20 each map to one test
 * function below; see the section banner above each.
 */

#include "decoder_core.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Assert macros (mirrors tools/segmenter_replay/test_segmenter_core.c)
// ---------------------------------------------------------------------------

#define ASSERT_EQ_INT(actual, expected, msg) do { \
    int _a = (int)(actual), _e = (int)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %d, expected %d\n", \
                __func__, (msg), _a, _e); \
        exit(1); \
    } \
} while (0)

#define ASSERT_GT_FLT(actual, bound, msg) do { \
    float _a = (float)(actual), _b = (float)(bound); \
    if (!(_a > _b)) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %f, expected > %f\n", \
                __func__, (msg), (double)_a, (double)_b); \
        exit(1); \
    } \
} while (0)

#define ASSERT_LT_FLT(actual, bound, msg) do { \
    float _a = (float)(actual), _b = (float)(bound); \
    if (!(_a < _b)) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %f, expected < %f\n", \
                __func__, (msg), (double)_a, (double)_b); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Letter-index + helpers
// ---------------------------------------------------------------------------

#define LX(ch) ((uint8_t)((ch) - 'A'))

// Build a top-K event: `letters` is a 5-char string of distinct A..Z, `probs`
// is the corresponding probability for each slot (in order). letters[0] is
// top-1; the rest are runners-up. Probabilities don't have to sum to 1 —
// remaining mass is split uniformly across the 21 outside-K letters per the
// PRD posterior-assembly spec.
static dec_top_k_t mk_top_k(const char *letters, const float *probs)
{
    dec_top_k_t t;
    memset(&t, 0, sizeof(t));
    for (int k = 0; k < SPELL_LETTER_TOP_K; k++) {
        t.candidates[k].letter_index = LX(letters[k]);
        t.candidates[k].probability  = probs[k];
    }
    return t;
}

static dec_word_t mk_word(uint32_t id, const char *str)
{
    dec_word_t w;
    memset(&w, 0, sizeof(w));
    w.word_id = id;
    int n = (int)strlen(str);
    if (n > SPELL_MAX_LETTERS_PER_WORD) n = SPELL_MAX_LETTERS_PER_WORD;
    w.length = (uint8_t)n;
    for (int i = 0; i < n; i++) w.letters[i] = LX(str[i]);
    return w;
}

// "Neutral" confusion matrix: M[j,j] = 0.6, off-diagonal = 0.4 / 25 ≈ 0.016
// per cell. Row sums = 1.0 exactly. No specific letter pair is privileged —
// useful as the default for tests that don't care about the matrix.
static dec_confusion_t conf_neutral(void)
{
    dec_confusion_t c;
    const float diag = 0.6f;
    const float off  = 0.4f / 25.0f;
    for (int j = 0; j < DEC_ALPHABET_SIZE; j++) {
        for (int i = 0; i < DEC_ALPHABET_SIZE; i++) {
            c.m[j * DEC_ALPHABET_SIZE + i] = (i == j) ? diag : off;
        }
    }
    return c;
}

// Bias the matrix so that when top-1 is `top1`, the true letter is often
// `true_letter`. P(true=true_letter | top1) = `pair_prob`. The remaining
// 1 - pair_prob mass is split uniformly across the other 25 letters
// (including the diagonal, so this overwrites M[j,j] too).
static void conf_inject_pair(dec_confusion_t *c,
                             char top1, char true_letter,
                             float pair_prob)
{
    int j = LX(top1), i = LX(true_letter);
    float others = 1.0f - pair_prob;
    if (others < 0.0f) others = 0.0f;
    float per_other = others / (float)(DEC_ALPHABET_SIZE - 1);
    for (int ii = 0; ii < DEC_ALPHABET_SIZE; ii++) {
        c->m[j * DEC_ALPHABET_SIZE + ii] = (ii == i) ? pair_prob : per_other;
    }
}

// ---------------------------------------------------------------------------
// Probability sets used across tests. Five-element arrays match
// SPELL_LETTER_TOP_K. Sums shown in comments.
// ---------------------------------------------------------------------------

// High-confidence top-1 (~0.85), runners-up negligible. Sum ≈ 0.99.
static const float P_HIGH[5] = { 0.85f, 0.05f, 0.04f, 0.03f, 0.02f };

// Wrong-top-1 case for confusion-pair recovery: top-1 has 0.5, top-2 has 0.3.
// Sum ≈ 0.95.
static const float P_WRONG_TOP1[5] = { 0.50f, 0.30f, 0.05f, 0.05f, 0.05f };

// Flat distribution: no clear winner. Sum = 0.75 (remaining 0.25 spreads
// across the 21 outside-K letters, putting them all near equal posterior).
static const float P_FLAT[5] = { 0.17f, 0.16f, 0.15f, 0.14f, 0.13f };

// Low-confidence (W fragments): top-1 only 0.30, distributed runners-up.
static const float P_LOW[5] = { 0.30f, 0.25f, 0.20f, 0.15f, 0.10f };

// ---------------------------------------------------------------------------
// Test #1 — Clean spelling resolves to the exact dictionary word.
// Acceptance: "clean spelling → exact resolution"
// ---------------------------------------------------------------------------
static void test_clean_spelling_resolves(void)
{
    dec_word_t words[] = {
        mk_word(1, "CAT"),
        mk_word(2, "BAT"),
        mk_word(3, "BAD"),
    };
    dec_dict_t dict = { .words = words, .n_words = 3 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    // Each top-K uses 4 distractor letters that don't appear in any dict word.
    dec_top_k_t tk_c = mk_top_k("CJKMP", P_HIGH);
    dec_top_k_t tk_a = mk_top_k("AJKMP", P_HIGH);
    dec_top_k_t tk_t = mk_top_k("TJKMP", P_HIGH);

    dec_on_letter(&st, &tk_c, 0);
    dec_on_letter(&st, &tk_a, 0);
    dec_on_letter(&st, &tk_t, 0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,    DEC_OUTCOME_RESOLVED, "clean CAT must resolve");
    ASSERT_EQ_INT(r.word_id, 1,                    "should resolve to CAT");
    ASSERT_GT_FLT(r.score_margin, cfg.decision_margin,
                  "clean spelling margin must clear decision threshold");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #2 — Confusion-pair recovery: top-1 wrong but the dictionary word
// containing the correct letter still wins, helped by the M[wrong, true]
// confusion-matrix entry shifting posterior mass back to the true letter.
// Acceptance: "one wrong top-1 with confusion-pair recovery"
// ---------------------------------------------------------------------------
static void test_confusion_pair_recovery(void)
{
    dec_word_t words[] = {
        mk_word(1, "BANK"),
        mk_word(2, "BARK"),  // distractor: differs from BANK at position 3
    };
    dec_dict_t dict = { .words = words, .n_words = 2 };

    // Strong M-with-N confusion: when top-1 = M, true is often N.
    dec_confusion_t conf = conf_neutral();
    conf_inject_pair(&conf, 'M', 'N', 0.30f);

    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    dec_top_k_t tk_b = mk_top_k("BJKPQ", P_HIGH);
    dec_top_k_t tk_a = mk_top_k("AJKPQ", P_HIGH);
    // Position 3: top-1 = M (wrong), top-2 = N (true). R is intentionally
    // outside this top-K so BARK's posterior mass at position 3 collapses
    // to the uniform-outside-K floor — the dictionary disambiguates BANK
    // from BARK by R's near-zero posterior, and the M→N confusion-matrix
    // entry boosts N's posterior on top of that.
    dec_top_k_t tk_m_wrong = mk_top_k("MNJPQ", P_WRONG_TOP1);
    dec_top_k_t tk_k = mk_top_k("KJPQX", P_HIGH);

    dec_on_letter(&st, &tk_b,       0);
    dec_on_letter(&st, &tk_a,       0);
    dec_on_letter(&st, &tk_m_wrong, 0);
    dec_on_letter(&st, &tk_k,       0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,    DEC_OUTCOME_RESOLVED,
                  "wrong top-1 must still resolve via dictionary");
    ASSERT_EQ_INT(r.word_id, 1, "should resolve to BANK, not BARK");
    ASSERT_GT_FLT(r.score_margin, cfg.decision_margin,
                  "BANK-vs-BARK margin must clear threshold");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #3 — 1-edit insertion: extra utterance (e.g. cough) gets aligned
// out by the ins_penalty path. n_positions = word_len + 1.
// Acceptance: "1-edit insertion"
// ---------------------------------------------------------------------------
static void test_one_edit_insertion(void)
{
    dec_word_t words[] = {
        mk_word(1, "CAT"),
        mk_word(2, "DOG"),  // length-3 distractor with no shared letters
    };
    dec_dict_t dict = { .words = words, .n_words = 2 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    // C, A, T as expected, then a spurious "X" utterance (a cough that the
    // model labelled X with high confidence). 4 utterances → 1-edit-insertion
    // path against length-3 dict words.
    dec_top_k_t tk_c     = mk_top_k("CJKMP", P_HIGH);
    dec_top_k_t tk_a     = mk_top_k("AJKMP", P_HIGH);
    dec_top_k_t tk_t     = mk_top_k("TJKMP", P_HIGH);
    dec_top_k_t tk_cough = mk_top_k("XJKMP", P_HIGH);

    dec_on_letter(&st, &tk_c,     0);
    dec_on_letter(&st, &tk_a,     0);
    dec_on_letter(&st, &tk_t,     0);
    dec_on_letter(&st, &tk_cough, 0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,    DEC_OUTCOME_RESOLVED,
                  "spurious utterance must not block resolution");
    ASSERT_EQ_INT(r.word_id, 1, "should resolve to CAT despite the cough");
    ASSERT_GT_FLT(r.score_margin, cfg.decision_margin,
                  "CAT margin must clear threshold even with ins_penalty");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #4 — 1-edit deletion: a missed letter (swallowed pause) gets aligned
// in by the del_penalty path. word_len = n_positions + 1.
// Acceptance: "1-edit deletion"
// ---------------------------------------------------------------------------
static void test_one_edit_deletion(void)
{
    dec_word_t words[] = {
        mk_word(1, "BANK"),
        mk_word(2, "BENT"),  // length-4 distractor that wouldn't align to B-A-K
    };
    dec_dict_t dict = { .words = words, .n_words = 2 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    // Spell B, A, K — the N was missed (swallowed pause). 3 utterances vs
    // length-4 dict words triggers the deletion path.
    dec_top_k_t tk_b = mk_top_k("BJKMP", P_HIGH);
    dec_top_k_t tk_a = mk_top_k("AJKMP", P_HIGH);
    dec_top_k_t tk_k = mk_top_k("KJPQX", P_HIGH);

    dec_on_letter(&st, &tk_b, 0);
    dec_on_letter(&st, &tk_a, 0);
    dec_on_letter(&st, &tk_k, 0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,    DEC_OUTCOME_RESOLVED,
                  "missed letter must still resolve via deletion path");
    ASSERT_EQ_INT(r.word_id, 1, "B-A-K should align to BANK (skip N)");
    ASSERT_GT_FLT(r.score_margin, cfg.decision_margin,
                  "BANK margin over BENT must clear threshold");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #5 — Flat distribution → abstain. Top-K probabilities are nearly
// uniform with no top-1 in any dictionary word; every candidate scores
// similarly and the runner-up margin is tiny.
// Acceptance: "abstain on flat distribution"
// ---------------------------------------------------------------------------
static void test_flat_distribution_abstains(void)
{
    dec_word_t words[] = {
        mk_word(1, "CAT"),
        mk_word(2, "BAT"),
        mk_word(3, "RAT"),
    };
    dec_dict_t dict = { .words = words, .n_words = 3 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    // Top-K uses letters not in any dict word — posterior over C/B/R/A/T is
    // dominated by the uniform-outside-K floor and the matrix smoothing.
    dec_top_k_t tk_flat = mk_top_k("JFXQH", P_FLAT);

    dec_on_letter(&st, &tk_flat, 0);
    dec_on_letter(&st, &tk_flat, 0);
    dec_on_letter(&st, &tk_flat, 0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind, DEC_OUTCOME_ABSTAIN,
                  "flat posteriors should abstain, not guess");
    ASSERT_LT_FLT(r.score_margin, cfg.decision_margin,
                  "margin should be below decision threshold");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #6 — Buffer overflow at MAX_LETTERS_PER_WORD → abstain (no crash).
// Acceptance: "MAX_LETTERS_PER_WORD overflow → abstain"
// ---------------------------------------------------------------------------
static void test_overflow_abstains(void)
{
    dec_word_t words[] = { mk_word(1, "CAT") };
    dec_dict_t dict = { .words = words, .n_words = 1 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    dec_top_k_t tk = mk_top_k("AJKMP", P_HIGH);

    // Submit MAX + 1 utterances. The (MAX+1)th must trip overflow.
    for (int i = 0; i <= SPELL_MAX_LETTERS_PER_WORD; i++) {
        dec_on_letter(&st, &tk, 0);
    }

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,       DEC_OUTCOME_ABSTAIN, "overflow must abstain");
    ASSERT_EQ_INT(r.overflowed, 1,                   "overflow flag must be set");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Test #7 — retract_count = 2 drops the two most recent in-flight entries
// and appends top_k. Models the W-recovery cycle's terminal event per
// ADR-0006: two low-confidence utterances are replaced by a single
// confirmed-W top-K. Includes a defensive sub-case where retract_count
// exceeds the in-flight count and must clamp.
// Acceptance: "LETTER_RECOGNIZED { retract_count = 2 } correctly drops
//             two prior entries and appends the W replacement"
// ---------------------------------------------------------------------------
static void test_retract_two_replaces(void)
{
    // "WAT" stands in for any W-leading word — exercises the retract path
    // through to a downstream resolution. Distractor SAT shares no W.
    dec_word_t words[] = {
        mk_word(1, "WAT"),
        mk_word(2, "SAT"),
    };
    dec_dict_t dict = { .words = words, .n_words = 2 };
    dec_confusion_t conf = conf_neutral();
    dec_config_t cfg; dec_config_default(&cfg);
    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "FAIL [%s]: dec_init rejected config\n", __func__);
        exit(1);
    }

    dec_top_k_t tk_lowD = mk_top_k("DUBOY", P_LOW);    // first W fragment
    dec_top_k_t tk_lowY = mk_top_k("YUOEI", P_LOW);    // second W fragment
    dec_top_k_t tk_w    = mk_top_k("WJKMP", P_HIGH);   // merged-rerun result
    dec_top_k_t tk_a    = mk_top_k("AJKMP", P_HIGH);
    dec_top_k_t tk_t    = mk_top_k("TJKMP", P_HIGH);

    // Two low-confidence utterances enter the buffer normally.
    dec_on_letter(&st, &tk_lowD, 0);
    dec_on_letter(&st, &tk_lowY, 0);
    ASSERT_EQ_INT(st.n_positions, 2, "should hold 2 positions before retract");

    // The W-recovery cycle's terminal event arrives: drop 2, append W.
    dec_on_letter(&st, &tk_w, 2);
    ASSERT_EQ_INT(st.n_positions, 1, "retract+append should leave 1 position");
    ASSERT_EQ_INT(st.positions[0].candidates[0].letter_index, LX('W'),
                  "position 0 top-1 must be the W replacement, not the first low-conf letter");

    // Continue with normal letters; should resolve to WAT.
    dec_on_letter(&st, &tk_a, 0);
    dec_on_letter(&st, &tk_t, 0);

    dec_resolution_t r = dec_resolve_full_eow(&st);
    ASSERT_EQ_INT(r.kind,    DEC_OUTCOME_RESOLVED, "WAT must resolve after W replacement");
    ASSERT_EQ_INT(r.word_id, 1, "should resolve to WAT");

    // Defensive sub-case: retract_count > n_positions clamps cleanly without
    // crashing or wrapping. From a fresh state with one position, a
    // retract=2 should clamp to 1 (drop the lone entry) and then append.
    dec_reset(&st);
    dec_on_letter(&st, &tk_a, 0);
    dec_on_letter(&st, &tk_w, 2);
    ASSERT_EQ_INT(st.n_positions, 1,
                  "retract_count > n_positions must clamp safely");
    ASSERT_EQ_INT(st.positions[0].candidates[0].letter_index, LX('W'),
                  "after clamped retract, position 0 must be the new top_k");

    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(void)
{
    test_clean_spelling_resolves();
    test_confusion_pair_recovery();
    test_one_edit_insertion();
    test_one_edit_deletion();
    test_flat_distribution_abstains();
    test_overflow_abstains();
    test_retract_two_replaces();

    printf("\nAll tests passed.\n");
    return 0;
}
