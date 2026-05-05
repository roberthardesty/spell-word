/**
 * @file test_w_detector_core.c
 * @brief Unit tests for the W-recovery trigger predicate.
 *
 * Synthesises sequences of (top-1 letter, top-1 probability) events and
 * asserts that the trigger fires iff all four conditions hold and the
 * sliding window has at least two prior events. No real audio, no IDF.
 *
 * Build + run:
 *   cd tools/w_detector_replay && make test
 *
 * Each test prints "PASS: <name>" or aborts with a detailed FAIL message.
 * Exit code 0 = all pass.
 *
 * The eight acceptance cases in Vikunja #28 each map to one or more test
 * functions below; see the section banner above each.
 */

#include "w_detector_core.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Assert macros (mirror tools/decoder_replay/test_decoder_core.c)
// ---------------------------------------------------------------------------

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s]: %s\n  expected true: %s\n", \
                __func__, (msg), #cond); \
        exit(1); \
    } \
} while (0)

#define ASSERT_FALSE(cond, msg) do { \
    if ((cond)) { \
        fprintf(stderr, "FAIL [%s]: %s\n  expected false: %s\n", \
                __func__, (msg), #cond); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Letter helpers (alphabet index 0='A')
// ---------------------------------------------------------------------------

#define LX(ch) ((uint8_t)((ch) - 'A'))

// Drive a single event through the predicate and return the eval.
static wdet_eval_t step(wdet_state_t *st, char letter, float prob)
{
    return wdet_on_event(st, LX(letter), prob);
}

// Initialize with defaults (matches spell_config.h: U=0.50, low=0.40).
static void init_default(wdet_state_t *st)
{
    wdet_config_t cfg;
    wdet_config_default(&cfg);
    int rc = wdet_init(st, &cfg);
    if (rc != 0) {
        fprintf(stderr, "FAIL [setup]: wdet_init returned %d\n", rc);
        exit(1);
    }
}

// =============================================================================
// Test 1 — three high-confidence non-W events: no trigger ever.
// =============================================================================

static void test_three_hi_conf_non_w_no_trigger(void)
{
    wdet_state_t st;
    init_default(&st);

    wdet_eval_t e1 = step(&st, 'A', 0.92f);
    ASSERT_FALSE(e1.trigger, "first event cannot trigger (no history)");
    ASSERT_FALSE(e1.sufficient_history, "first event has zero priors");

    wdet_eval_t e2 = step(&st, 'B', 0.88f);
    ASSERT_FALSE(e2.trigger, "second event still has only one prior");
    ASSERT_FALSE(e2.sufficient_history, "second event has one prior");

    wdet_eval_t e3 = step(&st, 'C', 0.95f);
    ASSERT_FALSE(e3.trigger, "current is not 'U' — no trigger");
    ASSERT_TRUE(e3.sufficient_history, "third event has two priors");
    ASSERT_FALSE(e3.current_is_u, "C is not U");
    ASSERT_FALSE(e3.prev1_low_conf, "B@0.88 is high-conf");
    ASSERT_FALSE(e3.prev2_low_conf, "A@0.92 is high-conf");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 2 — 2 lo-conf priors followed by hi-conf 'U': trigger fires.
// =============================================================================

static void test_two_lo_then_hi_u_triggers(void)
{
    wdet_state_t st;
    init_default(&st);

    step(&st, 'X', 0.30f);  // prev[2] (after we shift)
    step(&st, 'Y', 0.25f);  // prev[1]
    wdet_eval_t e = step(&st, 'U', 0.65f);

    ASSERT_TRUE(e.sufficient_history, "two priors present");
    ASSERT_TRUE(e.current_is_u, "current is U");
    ASSERT_TRUE(e.current_high_conf, "0.65 >= 0.50");
    ASSERT_TRUE(e.prev1_low_conf, "Y@0.25 < 0.40");
    ASSERT_TRUE(e.prev2_low_conf, "X@0.30 < 0.40");
    ASSERT_TRUE(e.trigger, "all four conditions satisfied");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 3 — 2 hi-conf priors followed by hi-conf 'U': no trigger
//         (priors fail the lo-conf condition, even though current is hi-conf U).
// =============================================================================

static void test_two_hi_then_hi_u_no_trigger(void)
{
    wdet_state_t st;
    init_default(&st);

    step(&st, 'P', 0.85f);
    step(&st, 'Q', 0.80f);
    wdet_eval_t e = step(&st, 'U', 0.70f);

    ASSERT_TRUE(e.current_is_u, "current is U");
    ASSERT_TRUE(e.current_high_conf, "0.70 >= 0.50");
    ASSERT_FALSE(e.prev1_low_conf, "Q@0.80 not lo-conf");
    ASSERT_FALSE(e.prev2_low_conf, "P@0.85 not lo-conf");
    ASSERT_FALSE(e.trigger, "priors are hi-conf");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 4 — 2 lo-conf priors followed by hi-conf NON-'U': no trigger.
// =============================================================================

static void test_two_lo_then_hi_non_u_no_trigger(void)
{
    wdet_state_t st;
    init_default(&st);

    step(&st, 'X', 0.20f);
    step(&st, 'Y', 0.30f);
    wdet_eval_t e = step(&st, 'A', 0.99f);

    ASSERT_FALSE(e.current_is_u, "A is not U");
    ASSERT_TRUE(e.current_high_conf, "0.99 >= 0.50 (still high-conf, but wrong letter)");
    ASSERT_TRUE(e.prev1_low_conf, "lo-conf prior");
    ASSERT_TRUE(e.prev2_low_conf, "lo-conf prior");
    ASSERT_FALSE(e.trigger, "current is not U");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 5 — 2 lo-conf priors followed by LO-conf 'U': no trigger
//         (current is U but below the U_THRESHOLD).
// =============================================================================

static void test_two_lo_then_lo_u_no_trigger(void)
{
    wdet_state_t st;
    init_default(&st);

    step(&st, 'X', 0.20f);
    step(&st, 'Y', 0.30f);
    wdet_eval_t e = step(&st, 'U', 0.35f);

    ASSERT_TRUE(e.current_is_u, "current is U");
    ASSERT_FALSE(e.current_high_conf, "0.35 < 0.50");
    ASSERT_TRUE(e.prev1_low_conf, "lo-conf prior");
    ASSERT_TRUE(e.prev2_low_conf, "lo-conf prior");
    ASSERT_FALSE(e.trigger, "current U below U_THRESHOLD");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 6 — threshold boundary cases.
//
// PRD spec: current_prob >= U_THRESHOLD  (>= is inclusive)
//           prev_top1_prob <  LOW_CONF_THRESHOLD  (< is strict)
//
// At U_THRESHOLD exactly: the U-condition holds.
// At LOW_CONF_THRESHOLD exactly: the lo-conf condition does NOT hold.
//
// We sweep both boundaries with custom thresholds so the test is robust to
// future tweaks in spell_config.h (the assertion is on the comparator
// semantics, not the specific default).
// =============================================================================

static void test_threshold_boundary(void)
{
    wdet_config_t cfg = { .u_threshold = 0.50f, .low_conf_threshold = 0.40f };
    wdet_state_t st;
    int rc = wdet_init(&st, &cfg);
    ASSERT_TRUE(rc == 0, "init succeeded");

    // (a) prev exactly AT the lo-conf boundary should fail (strict <).
    step(&st, 'X', 0.40f);  // prev[2] after shift — at boundary, NOT lo-conf
    step(&st, 'Y', 0.30f);  // prev[1] — clearly lo-conf
    wdet_eval_t e_a = step(&st, 'U', 0.55f);
    ASSERT_TRUE(e_a.current_high_conf, "0.55 above U threshold");
    ASSERT_TRUE(e_a.prev1_low_conf, "Y@0.30 lo-conf");
    ASSERT_FALSE(e_a.prev2_low_conf, "X@0.40 at boundary — strict <, so NOT lo-conf");
    ASSERT_FALSE(e_a.trigger, "prev[2] at boundary blocks trigger");

    // Reset and check (b) current exactly AT the U threshold fires (>=).
    wdet_reset(&st);
    step(&st, 'X', 0.20f);
    step(&st, 'Y', 0.20f);
    wdet_eval_t e_b = step(&st, 'U', 0.50f);
    ASSERT_TRUE(e_b.current_high_conf, "0.50 == U threshold counts as high-conf");
    ASSERT_TRUE(e_b.trigger, "boundary U fires (>= is inclusive)");

    // (c) current at U threshold but prev exactly at low_conf boundary blocks.
    wdet_reset(&st);
    step(&st, 'X', 0.20f);
    step(&st, 'Y', 0.40f);  // at lo-conf boundary — strict <, NOT lo-conf
    wdet_eval_t e_c = step(&st, 'U', 0.50f);
    ASSERT_TRUE(e_c.current_high_conf, "0.50 high-conf");
    ASSERT_FALSE(e_c.prev1_low_conf, "Y@0.40 at boundary blocks");
    ASSERT_FALSE(e_c.trigger, "boundary prev blocks trigger");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 7 — not-enough-history (only 1 prior event).
// =============================================================================

static void test_insufficient_history(void)
{
    wdet_state_t st;
    init_default(&st);

    // Only ONE lo-conf prior, then a perfect-shape 'U' — must NOT fire.
    step(&st, 'X', 0.20f);
    wdet_eval_t e = step(&st, 'U', 0.90f);

    ASSERT_FALSE(e.sufficient_history, "only one prior at evaluation time");
    ASSERT_FALSE(e.trigger, "insufficient history blocks trigger even with perfect U");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 8 — consecutive triggers (W-W back-to-back).
//
// After a successful W-trigger, two more lo-conf events followed by a hi-conf
// 'U' should ALSO fire. The sliding window keeps moving — w_detector_core
// does not auto-reset on trigger (the recognizer in #30 may explicitly
// wdet_reset() if it wants a fresh window after a confirmed merge, but that
// is not the trigger predicate's concern).
// =============================================================================

static void test_consecutive_triggers(void)
{
    wdet_state_t st;
    init_default(&st);

    // First W:
    step(&st, 'X', 0.25f);
    step(&st, 'Y', 0.30f);
    wdet_eval_t first = step(&st, 'U', 0.65f);
    ASSERT_TRUE(first.trigger, "first W triggers");

    // Two more lo-conf utterances seed the next window.
    wdet_eval_t mid_a = step(&st, 'X', 0.20f);
    wdet_eval_t mid_b = step(&st, 'Y', 0.30f);
    ASSERT_FALSE(mid_a.trigger, "first lo-conf after trigger has hi-conf prev (the U)");
    ASSERT_FALSE(mid_b.trigger, "still hi-conf prev[2] = U");

    // Second hi-conf U with two lo-conf priors → fires again.
    wdet_eval_t second = step(&st, 'U', 0.70f);
    ASSERT_TRUE(second.sufficient_history, "two priors in window");
    ASSERT_TRUE(second.current_is_u, "current is U");
    ASSERT_TRUE(second.current_high_conf, "high-conf U");
    ASSERT_TRUE(second.prev1_low_conf, "Y@0.30 lo-conf");
    ASSERT_TRUE(second.prev2_low_conf, "X@0.20 lo-conf");
    ASSERT_TRUE(second.trigger, "second W fires back-to-back");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 9 — wdet_reset() forces insufficient history again.
//
// Ensures the recognizer in #30 has a clean way to gate post-merge state if
// it wants to.
// =============================================================================

static void test_reset_drops_history(void)
{
    wdet_state_t st;
    init_default(&st);

    step(&st, 'X', 0.20f);
    step(&st, 'Y', 0.30f);
    wdet_reset(&st);

    // Right after reset, a hi-conf U with no priors must NOT fire.
    wdet_eval_t e = step(&st, 'U', 0.95f);
    ASSERT_FALSE(e.sufficient_history, "reset cleared priors");
    ASSERT_FALSE(e.trigger, "reset blocks immediate trigger");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Test 10 — diagnostic flags carry the current event values for logging.
// =============================================================================

static void test_eval_carries_current_event(void)
{
    wdet_state_t st;
    init_default(&st);

    wdet_eval_t e = step(&st, 'U', 0.73f);
    ASSERT_TRUE(e.current_top1 == LX('U'), "current_top1 carried in eval");
    // Allow a tiny FP wobble; we wrote 0.73 directly.
    float diff = e.current_top1_prob - 0.73f;
    if (diff < 0.0f) diff = -diff;
    ASSERT_TRUE(diff < 1e-6f, "current_top1_prob carried in eval");

    printf("PASS: %s\n", __func__);
}

// =============================================================================
// Main
// =============================================================================

int main(void)
{
    test_three_hi_conf_non_w_no_trigger();
    test_two_lo_then_hi_u_triggers();
    test_two_hi_then_hi_u_no_trigger();
    test_two_lo_then_hi_non_u_no_trigger();
    test_two_lo_then_lo_u_no_trigger();
    test_threshold_boundary();
    test_insufficient_history();
    test_consecutive_triggers();
    test_reset_drops_history();
    test_eval_carries_current_event();

    printf("\nAll tests passed.\n");
    return 0;
}
