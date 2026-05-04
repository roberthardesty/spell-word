/**
 * @file segmenter_core.h
 * @brief Pure-C, IDF-free letter VAD state machine.
 *
 * This file is the unit-of-test for VAD tuning. It compiles as plain C99 on
 * any host with no ESP-IDF, no FreeRTOS, no esp-dsp — only <stdint.h>,
 * <stdbool.h>, <math.h>, and <string.h>. The tools/segmenter_replay/ host
 * program links it directly to run the algorithm against recorded WAVs.
 *
 * The algorithm — full version per plan §3 Phase 3:
 *
 *   - Per-frame RMS in dBFS, smoothed via EMA (~3-frame time constant).
 *   - Onset/offset thresholds with hysteresis. SEG_IN_LETTER on smoothed
 *     RMS ≥ on_dbfs; transitions to SEG_POST_LETTER when smoothed RMS
 *     drops below off_dbfs. Confirmed letter offset = off_frames consecutive
 *     sub-off frames; until then, frames keep accumulating (handles brief
 *     dips inside a letter).
 *   - Pre-roll ring buffer holds the last preroll_samples of audio
 *     continuously, so an emitted utterance includes pre-onset context —
 *     the leading edge of the letter, which the network was trained on.
 *   - Min-letter duration: emissions shorter than min_letter_frames are
 *     rejected (not emitted, but EOW counter still progresses).
 *   - Max-letter duration: a letter that runs longer than max_letter_frames
 *     is force-split. The accumulated PCM is emitted as one utterance, the
 *     state machine returns to SEG_IN_LETTER (without re-arming pre-roll),
 *     and continues accumulating.
 *   - End-of-word: after a letter is emitted, eow_frames of continuous
 *     silence (no onset) without a new letter generates a SEG_EVT_EOW.
 *     Inter-letter silence (between letters within a word) does NOT
 *     trigger EOW; it just keeps the word_in_flight flag set.
 *   - Early-commit window: after a letter is emitted, the first time the
 *     post-letter silence count crosses early_commit_frames (without a new
 *     onset) generates a SEG_EVT_EARLY_COMMIT_WINDOW. The decoder uses this
 *     to evaluate its early-commit predicate without polling. Latched
 *     per-gap: fires exactly once per inter-letter silence cycle, never
 *     twice without an intervening onset. Ordering with EOW is guaranteed
 *     by seg_init validating early_commit_frames < eow_frames.
 *
 * Frame size and sample rate are configured at init() and fixed for the
 * lifetime of the state. seg_process_frame() expects exactly cfg.frame_samples
 * samples per call.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    // --- Audio format ---
    int   frame_samples;            ///< samples per process_frame call (e.g. 256)
    int   sample_rate;              ///< Hz (e.g. 16000)

    // --- Thresholds (dBFS) ---
    float on_dbfs;                  ///< letter onset (e.g. -38.0)
    float off_dbfs;                 ///< letter offset (e.g. -42.0)

    // --- Smoothing ---
    /// EMA coefficient * 100, integer for portability across builds.
    /// A value of 35 means: ema = 0.35 * frame_dbfs + 0.65 * prev_ema.
    /// Larger = less smoothing (more responsive); smaller = more smoothing.
    /// 35 ≈ 3-frame time constant.
    int   ema_alpha_x100;

    // --- Timing (in frames; one frame = frame_samples / sample_rate seconds) ---
    int   off_frames;               ///< sub-off frames to confirm offset (~5 = 80 ms)
    int   min_letter_frames;        ///< reject letters shorter than this (~10 = 160 ms)
    int   max_letter_frames;        ///< force-split letters longer than this (~50 = 800 ms)
    int   early_commit_frames;      ///< inter-letter silence frames to fire early-commit window (~31 = 500 ms)
    int   eow_frames;               ///< silence frames to declare end-of-word (~75 = 1200 ms)

    // --- Buffers (caller-allocated, lifetime ≥ state) ---
    int16_t *preroll;               ///< ring buffer for pre-onset context
    int      preroll_samples;       ///< capacity of preroll[], in samples (~3200 = 200 ms)

    int16_t *accum;                 ///< accumulator for current letter (incl. pre-roll)
    int      accum_capacity;        ///< capacity of accum[], in samples
} seg_config_t;

// =============================================================================
// State
// =============================================================================

typedef enum {
    SEG_STATE_IDLE = 0,             ///< no letter in progress
    SEG_STATE_IN_LETTER,            ///< above onset, accumulating
    SEG_STATE_POST_LETTER,          ///< briefly below offset; confirming
} seg_state_kind_t;

typedef struct {
    seg_config_t cfg;

    seg_state_kind_t state;
    bool             word_in_flight;        ///< saw an emitted letter since last EOW
    bool             early_commit_window_fired; ///< latch: per-gap, reset on next onset
    int              silent_frames_offset;  ///< for offset confirmation
    int              silent_frames_eow;     ///< for end-of-word countdown (also drives early-commit window)
    int              letter_frames;         ///< frames accumulated in current letter
    float            ema_dbfs;              ///< smoothed RMS

    int              preroll_write_pos;     ///< next preroll write index (0..preroll_samples)
    int              preroll_filled;        ///< samples actually written so far (<= preroll_samples)
    int              accum_len;             ///< samples in accum[]
    int              letter_onset_in_accum; ///< sample offset within accum[] of letter onset
    float            letter_peak_dbfs;      ///< max EMA-smoothed RMS during current letter

    uint32_t         total_letters_emitted;
    uint32_t         total_letters_rejected_short;
    uint32_t         total_words_completed;
    uint32_t         total_force_splits;
} seg_state_t;

// =============================================================================
// Events returned by seg_process_frame()
// =============================================================================

typedef enum {
    SEG_EVT_NONE = 0,               ///< no transition this frame
    SEG_EVT_LETTER_EMITTED,         ///< utterance ready in cfg.accum (samples 0..n_samples)
    SEG_EVT_LETTER_REJECTED_SHORT,  ///< letter completed but below min_letter_frames; not emitted
    SEG_EVT_END_OF_WORD,            ///< sustained silence after at least one emitted letter
    SEG_EVT_FORCE_SPLIT,            ///< max_letter_frames reached; same as LETTER_EMITTED but flagged
    SEG_EVT_EARLY_COMMIT_WINDOW,    ///< inter-letter silence crossed early_commit_frames (latch: once per gap)
} seg_event_kind_t;

typedef struct {
    seg_event_kind_t kind;

    // Valid when kind ∈ {LETTER_EMITTED, FORCE_SPLIT}:
    /// Pointer into cfg.accum holding the utterance. Valid until the next
    /// seg_process_frame() call.
    const int16_t *pcm;
    int            n_samples;
    int            duration_ms;     ///< from onset to confirmed offset
    float          peak_dbfs;       ///< maximum smoothed RMS during the letter

    // Valid when kind == END_OF_WORD:
    uint32_t       letters_in_word; ///< how many letters were emitted in this word
} seg_event_t;

// =============================================================================
// API
// =============================================================================

/**
 * Initialize the state machine. The caller-allocated preroll and accum
 * buffers in @p cfg must remain valid for the lifetime of @p state.
 *
 * Returns 0 on success, -1 if the config is invalid (zero/negative sizes,
 * NULL buffers, off_frames > max_letter_frames, etc.).
 */
int seg_init(seg_state_t *state, const seg_config_t *cfg);

/**
 * Reset the state machine to IDLE. Pre-roll content is kept (it's continuously
 * updated regardless of state); accumulator and counters are cleared.
 *
 * Stat counters (total_letters_emitted, etc.) are NOT cleared — use
 * seg_reset_stats() for that.
 */
void seg_reset(seg_state_t *state);

/**
 * Reset stat counters to zero. Independent of seg_reset().
 */
void seg_reset_stats(seg_state_t *state);

/**
 * Process exactly @p n samples of PCM (must equal cfg.frame_samples).
 *
 * Returns a single event per frame. If multiple transitions could fire on
 * the same frame (e.g. a force-split coincident with EOW), only the
 * higher-priority one is reported and the rest are reset/queued so the
 * caller sees a clean sequence.
 *
 * The event's pcm pointer (when valid) points into cfg.accum and is
 * stable until the next seg_process_frame() call. Copy it out before
 * the next call.
 */
seg_event_t seg_process_frame(seg_state_t *state, const int16_t *frame, int n);

// =============================================================================
// Helpers (exposed for unit tests)
// =============================================================================

/**
 * Compute RMS of a frame in dBFS, normalized to int16 full-scale.
 *
 * Returns -100.0f for purely-silent frames (sum_sq < 1e-20).
 */
float seg_frame_rms_dbfs(const int16_t *frame, int n);

#ifdef __cplusplus
}  // extern "C"
#endif
