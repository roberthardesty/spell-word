/**
 * @file spell_events.h
 * @brief Cross-component esp_event types — event bases, IDs, and payloads.
 *
 * Components that emit events declare the matching ESP_EVENT_DEFINE_BASE in
 * their .c/.cpp file. Consumers register handlers via:
 *
 *   esp_event_handler_register(SPELL_INFERENCE_EVENT,
 *                              SPELL_EVENT_LETTER_TOP_K,
 *                              handler, ctx);
 */

#pragma once

#include <stdint.h>
#include "esp_event.h"
#include "spell_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Inference events — emitted by letter_classifier
// =============================================================================

ESP_EVENT_DECLARE_BASE(SPELL_INFERENCE_EVENT);

enum {
    /// Top-K letter candidates produced by the DS-CNN for one utterance.
    /// Payload: spell_letter_topk_event_t.
    SPELL_EVENT_LETTER_TOP_K = 1,
};

typedef struct {
    uint8_t letter_index;   ///< 0..25 → 'A'..'Z'
    float   probability;    ///< softmax probability in [0.0, 1.0]
} spell_letter_candidate_t;

typedef struct {
    spell_letter_candidate_t top_k[SPELL_LETTER_TOP_K];
    uint32_t                 invoke_ms;     ///< TFLM Invoke() wall-clock ms
} spell_letter_topk_event_t;

// =============================================================================
// Segmenter events — emitted by segmenter
// =============================================================================

ESP_EVENT_DECLARE_BASE(SPELL_SEGMENTER_EVENT);

enum {
    /// Sustained silence detected after a sequence of letters.
    /// Payload: spell_eow_event_t.
    SPELL_EVENT_END_OF_WORD = 1,

    /// User cancelled or armed timeout fired with no audio collected.
    /// Empty payload.
    SPELL_EVENT_SEGMENTER_RESET = 2,

    /// Inter-letter silence has crossed the early-commit threshold once in
    /// this gap. Latch resets on next onset. Decoder evaluates the
    /// early-commit predicate when it sees this. Empty payload.
    SPELL_EVENT_EARLY_COMMIT_WINDOW = 3,
};

typedef struct {
    uint32_t letter_count;  ///< how many utterances were emitted in this word
} spell_eow_event_t;

// =============================================================================
// Decoder events — emitted by decoder
// =============================================================================
//
// Each base is *declared* here so other components can register handlers
// against it. The matching ESP_EVENT_DEFINE_BASE lands in the producing
// component's .c file when that slice ships — until then, handler-register
// against the base will fail to link, which is the desired forcing function
// for sequencing.

ESP_EVENT_DECLARE_BASE(SPELL_DECODER_EVENT);

enum {
    /// A word was resolved (early-commit or full-EOW). Payload:
    /// spell_word_resolved_event_t.
    SPELL_EVENT_WORD_RESOLVED = 1,

    /// The decoder declined to commit (margin too tight, overflow, etc.).
    /// Empty payload.
    SPELL_EVENT_WORD_ABSTAIN = 2,
};

/// Why the decoder committed when it did. Carried in WORD_RESOLVED so the
/// per-spelling summary log line (US-25) can record commit provenance.
typedef enum {
    SPELL_COMMIT_REASON_EARLY = 0,    ///< early-commit predicate fired
    SPELL_COMMIT_REASON_FULL_EOW = 1, ///< full end-of-word silence reached
} spell_commit_reason_t;

typedef struct {
    uint32_t              word_id;       ///< dictionary index of resolved word
    spell_commit_reason_t commit_reason;
    float                 score_margin;  ///< score(best) - score(runner_up)
} spell_word_resolved_event_t;

// =============================================================================
// Playback events — emitted by spell_ui's playback worker
// =============================================================================

ESP_EVENT_DECLARE_BASE(SPELL_PLAYBACK_EVENT);

enum {
    /// Playback (tone or word) finished naturally. Cancelled playback does
    /// not emit COMPLETE. Payload: spell_playback_complete_event_t.
    SPELL_EVENT_PLAYBACK_COMPLETE = 1,
};

typedef struct {
    uint8_t was_word;   ///< 1 if play_word completed, 0 if a tone
} spell_playback_complete_event_t;

// =============================================================================
// UI events — emitted by spell_ui state machine
// =============================================================================

ESP_EVENT_DECLARE_BASE(SPELL_UI_EVENT);

/// Top-level UI states (PRD 0001 §"UX module"). Encoded here so transition
/// events can be carried as data and host-side ui_core tests can assert on
/// (state, event) pairs without dragging IDF in.
typedef enum {
    SPELL_UI_STATE_IDLE      = 0,
    SPELL_UI_STATE_ARMED     = 1,
    SPELL_UI_STATE_SPELLING  = 2,
    SPELL_UI_STATE_RESOLVING = 3,
    SPELL_UI_STATE_PLAYING   = 4,
} spell_ui_state_t;

enum {
    /// State machine transitioned. Payload: spell_ui_state_transition_event_t.
    SPELL_EVENT_UI_STATE_TRANSITION = 1,
};

typedef struct {
    spell_ui_state_t from_state;
    spell_ui_state_t to_state;
} spell_ui_state_transition_event_t;

// =============================================================================
// W-detector events — emitted by w_detector
// =============================================================================

ESP_EVENT_DECLARE_BASE(SPELL_W_DETECTOR_EVENT);

enum {
    /// Tells the decoder to pop the most recent retract_count top-K events
    /// from its in-flight buffer and push the replacement event in their
    /// place (PRD 0001 §"Decoder retract-and-replace"). Payload:
    /// spell_retract_and_replace_event_t.
    SPELL_EVENT_RETRACT_AND_REPLACE = 1,
};

typedef struct {
    uint8_t                   retract_count; ///< usually 3 (the W trigger pattern)
    spell_letter_topk_event_t replacement;   ///< re-inference top-K (a confident W)
} spell_retract_and_replace_event_t;

#ifdef __cplusplus
}  // extern "C"
#endif
