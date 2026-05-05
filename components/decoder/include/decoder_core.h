/**
 * @file decoder_core.h
 * @brief Pure-C, IDF-free word decoder.
 *
 * Compiles as plain C99 on any host with no ESP-IDF, no FreeRTOS — only
 * <stdbool.h>, <stdint.h>, <math.h>, <string.h>. The tools/decoder_replay/
 * host program links it directly to run the algorithm against synthetic
 * top-K event fixtures. The IDF wrapper (separate slice — Vikunja #22)
 * adds esp_event subscription and partition loading.
 *
 * Algorithm — full version per PRD docs/prds/0001-phase-4-5-ux-and-decoder.md
 * §"Posterior assembly" and §"Decoder letter-event handler (unified)":
 *
 *   - On each LETTER_RECOGNIZED event, dec_on_letter() pops `retract_count`
 *     entries from the in-flight buffer, then appends top_k. retract_count=0
 *     is the no-op fast path (every clean utterance); retract_count=2 fires
 *     only when the recognizer's W-recovery cycle confirms a multi-utterance
 *     W (per ADR-0005, ADR-0006). One code path; no special "retract" branch.
 *
 *   - On full end-of-word, dec_resolve_full_eow() assembles per-letter
 *     posteriors (top-K + uniform prior over the 21 outside-K letters,
 *     blended with M[top1, *] at weight α), then scores every dictionary
 *     word under three alignment strategies — exact length, one inserted
 *     utterance (cough), one missed letter (swallowed pause) — and either
 *     returns the best word or abstains when the score margin is below
 *     SPELL_DECISION_MARGIN.
 *
 *   - The aggressive early-commit predicate (PRD §"Aggressive early-commit
 *     predicate", ADR-0001) is dec_try_early_commit(). It is a pure read-only
 *     function over the same in-flight state, so the caller can run it on
 *     every LETTER_RECOGNIZED event and every EARLY_COMMIT_WINDOW event and
 *     decide whether to publish WORD_RESOLVED. The post-letter silence floor
 *     (third clause) is supplied by the caller as a boolean: false on letter
 *     events, true on window events. decoder_core stays time-agnostic.
 *
 * The state struct holds no dynamic allocation — the in-flight buffer is a
 * fixed array sized to SPELL_MAX_LETTERS_PER_WORD. Dictionary and confusion
 * matrix are caller-owned; their lifetimes must equal the state's.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "spell_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEC_ALPHABET_SIZE 26

// =============================================================================
// Wire types — mirror spell_events.h shapes but stay IDF-free.
// =============================================================================
//
// The IDF wrapper memcpys spell_letter_top_k_t into dec_top_k_t. The two
// have identical layout: SPELL_LETTER_TOP_K candidates each {uint8_t,float}
// plus a uint32_t invoke_ms. We don't include esp_event.h here; the wrapper
// owns the conversion.

typedef struct {
    uint8_t letter_index;   ///< 0..25 → 'A'..'Z'
    float   probability;    ///< softmax probability in [0.0, 1.0]
} dec_candidate_t;

typedef struct {
    dec_candidate_t candidates[SPELL_LETTER_TOP_K];
    uint32_t        invoke_ms;
} dec_top_k_t;

// =============================================================================
// Dictionary + confusion matrix (caller-owned, immutable for state lifetime)
// =============================================================================

typedef struct {
    uint32_t word_id;                              ///< stable index used by playback
    uint8_t  length;                               ///< letters used in `letters`
    uint8_t  letters[SPELL_MAX_LETTERS_PER_WORD];  ///< 0..25 per slot
} dec_word_t;

typedef struct {
    const dec_word_t *words;
    int               n_words;
} dec_dict_t;

/// Row-major confusion matrix M[top1 * 26 + true_letter] = P(true | top1).
/// Loaded from the `matrix` partition in production; synthesised in tests.
typedef struct {
    float m[DEC_ALPHABET_SIZE * DEC_ALPHABET_SIZE];
} dec_confusion_t;

// =============================================================================
// Tunables (mirror spell_config.h; defaulted via dec_config_default())
// =============================================================================

typedef struct {
    float alpha;                 ///< SPELL_DECODER_ALPHA — confusion-matrix mix weight
    float decision_margin;       ///< SPELL_DECISION_MARGIN — abstain when below (full EOW)
    float ins_penalty;           ///< SPELL_EDIT_INS_PENALTY — utterance was spurious (cough)
    float del_penalty;           ///< SPELL_EDIT_DEL_PENALTY — letter was missed (swallowed pause)
    float early_margin_runnerup; ///< SPELL_EARLY_MARGIN_RUNNERUP — clause 1 threshold
    float early_margin_longer;   ///< SPELL_EARLY_MARGIN_LONGER  — clause 2 threshold
} dec_config_t;

/// Fill @p cfg with the spell_config.h defaults.
void dec_config_default(dec_config_t *cfg);

// =============================================================================
// State + outcome
// =============================================================================

typedef struct {
    const dec_dict_t      *dict;
    const dec_confusion_t *confusion;
    dec_config_t           cfg;

    dec_top_k_t positions[SPELL_MAX_LETTERS_PER_WORD];
    int         n_positions;
    /// Set when an append would have exceeded SPELL_MAX_LETTERS_PER_WORD.
    /// dec_resolve_full_eow() honours this by abstaining instead of crashing.
    bool        overflowed;
} dec_state_t;

typedef enum {
    DEC_OUTCOME_RESOLVED = 0,
    DEC_OUTCOME_ABSTAIN  = 1,
} dec_outcome_kind_t;

typedef struct {
    dec_outcome_kind_t kind;
    uint32_t           word_id;       ///< valid iff kind == RESOLVED
    float              score;         ///< best-aligned log-score
    float              score_margin;  ///< score(best) - score(runner_up); +INF if no runner-up
    int                n_positions;   ///< utterances seen at EOW (informational)
    bool               overflowed;    ///< true if abstain reason was buffer overflow
} dec_resolution_t;

// =============================================================================
// API
// =============================================================================

/**
 * Initialize @p st. The dict and confusion pointers must remain valid for
 * the lifetime of @p st; the cfg is copied. Returns 0 on success, -1 if any
 * pointer is NULL or the confusion matrix has obviously-bad rows
 * (sum < 0.5 or > 1.5 — strict 1.0 ± 1e-3 validation lives in the IDF
 * partition loader, not here, since host tests intentionally use coarse
 * matrices).
 */
int dec_init(dec_state_t *st,
             const dec_dict_t *dict,
             const dec_confusion_t *confusion,
             const dec_config_t *cfg);

/**
 * Reset the in-flight buffer. Dict / confusion / cfg are kept.
 */
void dec_reset(dec_state_t *st);

/**
 * Unified letter-event handler. Pops the most recent @p retract_count
 * entries from the in-flight buffer (no-op when retract_count = 0, the
 * common case), then appends @p top_k. If the append would exceed
 * SPELL_MAX_LETTERS_PER_WORD, sets the overflow flag (no crash). If
 * retract_count exceeds the in-flight count, it is clamped to that count
 * (defensive — should never happen in practice).
 */
void dec_on_letter(dec_state_t *st,
                   const dec_top_k_t *top_k,
                   uint8_t retract_count);

/**
 * Resolve the in-flight letter sequence against the dictionary. Returns
 * RESOLVED with the chosen word_id and score margin, or ABSTAIN when the
 * margin is below SPELL_DECISION_MARGIN, when the buffer overflowed during
 * collection, or when the dictionary is empty / no length-±1 alignment
 * exists. Pure function over @p st.
 */
dec_resolution_t dec_resolve_full_eow(const dec_state_t *st);

// =============================================================================
// Aggressive early-commit predicate (PRD §"Aggressive early-commit predicate",
// ADR-0001). Caller (IDF wrapper) evaluates on every LETTER_RECOGNIZED event
// (silence_floor_satisfied = false) and every EARLY_COMMIT_WINDOW event
// (silence_floor_satisfied = true) and publishes WORD_RESOLVED on
// DEC_EARLY_COMMIT_RESOLVED. Per-clause flags drive US-13 diagnostic logging.
// =============================================================================

typedef enum {
    DEC_EARLY_COMMIT_HOLD     = 0,   ///< at least one clause failed; do not commit
    DEC_EARLY_COMMIT_RESOLVED = 1,   ///< all three clauses passed; commit now
} dec_early_commit_kind_t;

typedef struct {
    dec_early_commit_kind_t kind;

    /// Clause 1: margin over same-length runner-up
    bool  margin_runnerup_ok;
    float margin_runnerup;     ///< score(best) - score(runner_up_same_length); +INF if no runner-up
    /// Clause 2: margin over best length-(N+1) or length-(N+2) candidate
    bool  margin_longer_ok;
    float margin_longer;       ///< score(best) - score(best_longer); +INF if no longer competitor
    /// Clause 3: post-letter silence floor (caller-supplied)
    bool  silence_floor_ok;

    /// Best same-length candidate. word_id valid iff kind == RESOLVED.
    uint32_t word_id;
    float    score;            ///< best same-length score
    int      n_positions;      ///< utterance count at evaluation time (informational)
} dec_early_commit_eval_t;

/**
 * Evaluate the three-clause early-commit predicate. Pure function over
 * @p st; never mutates state. The caller decides what to do with the result
 * (publish WORD_RESOLVED and dec_reset() on RESOLVED; log per-clause flags
 * on HOLD).
 *
 * The predicate considers same-length dictionary words (no edits) for the
 * "best" and "runner-up" pool, and length-(N+1) / length-(N+2) words via
 * the deletion-penalty path for the "longer competitor" pool. Insertion
 * alignments (shorter-than-N) are intentionally not considered at early
 * commit — the full-EOW resolver picks those up on dec_resolve_full_eow().
 *
 * On overflow, n_positions == 0, or empty dictionary, returns kind = HOLD
 * with all clause flags false (no commit). Computed margins use +INF when
 * the relevant competitor pool is empty (vacuously satisfies the clause).
 *
 * @param silence_floor_satisfied  true on EARLY_COMMIT_WINDOW events; false
 *                                 on LETTER_RECOGNIZED events. The third
 *                                 clause is load-bearing (ADR-0001) — it
 *                                 prevents prefix-truncation (BAN inside
 *                                 BANK) from committing mid-word.
 */
dec_early_commit_eval_t dec_try_early_commit(const dec_state_t *st,
                                             bool silence_floor_satisfied);

#ifdef __cplusplus
}  // extern "C"
#endif
