/**
 * @file w_detector_core.h
 * @brief Pure-C trigger predicate for the W-recovery cycle.
 *
 * Sub-module of letter_recognizer (per ADR-0005 — W-recovery lives inside
 * the recognizer; this trigger predicate is split out so it can be host-
 * tested in isolation against synthetic top-K event sequences).
 *
 * Compiles as plain C99 with no ESP-IDF, no FreeRTOS. The recognizer task
 * (C++) calls wdet_on_event() on every fresh top-K and acts on the eval:
 * if `trigger` is set, hold the 'U' emission and concatenate the last three
 * PCM buffers from the recognizer's PCM ring (Vikunja #27) for a single
 * re-inference pass. Whether the merged result is confirmed as 'W' or
 * rejected is the recognizer's concern (ADR-0005, Vikunja #30) — this
 * sub-module only fires the predicate.
 *
 * Predicate (PRD docs/prds/0001-phase-4-5-ux-and-decoder.md
 *           §"W-detection trigger"):
 *
 *   trigger ⇔  current.top1 == 'U'
 *           && current.top1_prob >= SPELL_W_DETECT_U_THRESHOLD
 *           && prev[1].top1_prob <  SPELL_W_DETECT_LOW_CONF_THRESHOLD
 *           && prev[2].top1_prob <  SPELL_W_DETECT_LOW_CONF_THRESHOLD
 *
 * `prev[1]` is the immediately-prior utterance, `prev[2]` the one before
 * that. Until two prior utterances exist (events_seen < 2 going into the
 * call), the trigger is unconditionally HOLD.
 *
 * Every evaluation populates per-condition flags (see wdet_eval_t) so the
 * recognizer's logging path can attribute false negatives and positives
 * (US-16). Pure function over @p st modulo the post-eval window shift —
 * the new event's top1_prob is pushed into the sliding window AFTER the
 * predicate is computed against the prior two.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "spell_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Number of PRIOR utterances tracked by the trigger. The "current" event
/// is the new argument to wdet_on_event(), not stored in this window.
#define WDET_HISTORY_DEPTH 2

/// 'U' encoded as alphabet index (0='A', 'U' = 20).
#define WDET_LETTER_U      20

// =============================================================================
// Tunables (mirror spell_config.h; defaulted via wdet_config_default())
// =============================================================================

typedef struct {
    /// SPELL_W_DETECT_U_THRESHOLD — current top-1 'U' must clear this (>=).
    float u_threshold;
    /// SPELL_W_DETECT_LOW_CONF_THRESHOLD — each prior top-1 prob below this (<).
    float low_conf_threshold;
} wdet_config_t;

/// Fill @p cfg with the spell_config.h defaults.
void wdet_config_default(wdet_config_t *cfg);

// =============================================================================
// State + eval result
// =============================================================================

typedef struct {
    wdet_config_t cfg;

    /// Sliding window of the last two prior events' top-1 probabilities.
    /// prev_top1_prob[0] = prev[1] (most recent), prev_top1_prob[1] = prev[2].
    float prev_top1_prob[WDET_HISTORY_DEPTH];

    /// Number of prior events stored. Saturates at WDET_HISTORY_DEPTH.
    uint8_t prior_count;
} wdet_state_t;

typedef struct {
    /// True iff all four predicate conditions hold (and history is sufficient).
    bool trigger;

    // Per-condition diagnostic flags — true means the condition was satisfied.
    bool current_is_u;
    bool current_high_conf;     ///< current_prob >= u_threshold
    bool prev1_low_conf;        ///< prev[1].prob  <  low_conf_threshold
    bool prev2_low_conf;        ///< prev[2].prob  <  low_conf_threshold
    bool sufficient_history;    ///< prior_count was >= 2 at evaluation time

    /// Carried through for logging — the values that drove the decision.
    uint8_t current_top1;
    float   current_top1_prob;
} wdet_eval_t;

// =============================================================================
// API
// =============================================================================

/**
 * Initialize @p st. The cfg is copied; @p cfg may be NULL to use defaults.
 * Returns 0 on success, -1 if @p st is NULL or thresholds are out of range
 * ([0.0, 1.0]).
 */
int wdet_init(wdet_state_t *st, const wdet_config_t *cfg);

/**
 * Reset the sliding window. Cfg is preserved. Use after a successful
 * W-recovery merge if the recognizer wants to start fresh — otherwise let
 * the window keep sliding so consecutive 'W' triggers can fire.
 */
void wdet_reset(wdet_state_t *st);

/**
 * Evaluate the trigger predicate against the new event, then shift it into
 * the sliding window. Order matters: the predicate is computed against the
 * TWO PRIOR events; only after the eval is the new event's top1_prob
 * pushed onto the window (so `prev[2]` falls off, `prev[1]` becomes the
 * new `prev[2]`, the new event becomes the new `prev[1]`).
 *
 * @param top1_index   alphabet index 0..25 of the new event's top-1 letter.
 * @param top1_prob    softmax probability of the new event's top-1 in [0,1].
 * @return wdet_eval_t with `trigger` set iff all four conditions hold.
 *         Per-condition flags are always populated for logging.
 */
wdet_eval_t wdet_on_event(wdet_state_t *st, uint8_t top1_index, float top1_prob);

#ifdef __cplusplus
}  // extern "C"
#endif
