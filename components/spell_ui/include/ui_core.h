/**
 * @file ui_core.h
 * @brief Pure-C UI state-machine transition table for the spell_ui module.
 *
 * Compiles as plain C99 on any host with no ESP-IDF, no FreeRTOS, no
 * esp_event — only <stdint.h>, <stdbool.h>, <string.h>. The
 * tools/ui_core_replay/ host program links it directly for unit tests.
 *
 * Per PRD 0001 §"Modules" and US-20: ui_core is a pure transition table
 * over (IDLE, ARMED, SPELLING, RESOLVING, PLAYING). For each (state, event)
 * pair, ui_core_step() returns (next_state, action_list). Every cell is
 * enumerated explicitly — no default branch that would silently swallow a
 * missing handler.
 *
 * Actions are *data*, not function pointers: the IDF wrapper consumes the
 * returned action_list and materializes each one (`PLAY_TONE`,
 * `CANCEL_PLAYBACK`, `SET_SEGMENTER_ACTIVE`, …) via the relevant IDF API.
 * This keeps the transition table host-testable — the tests assert against
 * the action-list payload without dragging FreeRTOS or esp_event in.
 *
 * Event vocabulary (translated by the IDF wrapper from the wire events):
 *
 *   BUTTON_PRESSED          : debounced button-press
 *   ARM_TIMEOUT             : arm-timer expired in ARMED with no audio
 *   FIRST_LETTER_ONSET      : first LETTER_RECOGNIZED of a spelling session
 *   END_OF_WORD             : segmenter EOW
 *   WORD_RESOLVED           : decoder resolved a word; arg = word_id
 *   WORD_ABSTAIN            : decoder declined to commit
 *   TONE_COMPLETE           : a tone (confirmation / resolved / error chirp)
 *                             finished playback
 *   WORD_PLAYBACK_COMPLETE  : a word playback finished
 *
 * The "early-commit-window with predicate-met" path from the PRD shows up
 * here as `WORD_RESOLVED` arriving while we are still in SPELLING — the
 * decoder commits early, we move to RESOLVING and play the resolved tone.
 * `END_OF_WORD` from SPELLING also goes to RESOLVING but without the tone
 * yet; the tone fires when WORD_RESOLVED arrives in RESOLVING.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "spell_events.h"   // spell_ui_state_t

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Events
// =============================================================================

typedef enum {
    SPELL_UI_EVT_BUTTON_PRESSED = 0,
    SPELL_UI_EVT_ARM_TIMEOUT,
    SPELL_UI_EVT_FIRST_LETTER_ONSET,
    SPELL_UI_EVT_END_OF_WORD,
    SPELL_UI_EVT_WORD_RESOLVED,            ///< arg.word_id valid
    SPELL_UI_EVT_WORD_ABSTAIN,
    SPELL_UI_EVT_TONE_COMPLETE,
    SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE,

    SPELL_UI_EVT_KIND_COUNT,
} spell_ui_event_kind_t;

typedef struct {
    spell_ui_event_kind_t kind;
    /// Carried with WORD_RESOLVED. Memoized internally so the eventual
    /// PLAY_WORD action (emitted on TONE_COMPLETE in RESOLVING) carries it.
    uint32_t              word_id;
} spell_ui_event_t;

// =============================================================================
// Actions
// =============================================================================

typedef enum {
    SPELL_UI_ACT_PLAY_TONE = 0,            ///< payload.tone_id ∈ SPELL_TONE_*
    SPELL_UI_ACT_PLAY_WORD,                ///< payload.word_id (last resolved)
    SPELL_UI_ACT_CANCEL_PLAYBACK,
    SPELL_UI_ACT_SET_SEGMENTER_ACTIVE,     ///< payload.seg_active
    SPELL_UI_ACT_START_ARM_TIMEOUT,
    SPELL_UI_ACT_CANCEL_ARM_TIMEOUT,

    SPELL_UI_ACT_KIND_COUNT,
} spell_ui_action_kind_t;

typedef struct {
    spell_ui_action_kind_t kind;
    union {
        uint8_t  tone_id;       ///< SPELL_TONE_CONFIRMATION / RESOLVED / ERROR_CHIRP
        uint32_t word_id;
        bool     seg_active;
    } payload;
} spell_ui_action_t;

/// Cap on actions per step. The fattest cell is PLAYING + BUTTON_PRESSED:
/// CANCEL_PLAYBACK, PLAY_TONE(confirmation), START_ARM_TIMEOUT — three
/// actions. Cap at 4 for headroom; a static_assert at translation time
/// would catch overflow but the table is hand-written, so we just trip
/// an internal assert if a future row exceeds this.
#define SPELL_UI_MAX_ACTIONS_PER_STEP   4

typedef struct {
    spell_ui_state_t  next_state;
    int               n_actions;
    spell_ui_action_t actions[SPELL_UI_MAX_ACTIONS_PER_STEP];
} spell_ui_step_t;

// =============================================================================
// State container
// =============================================================================

typedef struct {
    spell_ui_state_t state;
    /// Memoized from the most recent WORD_RESOLVED so the eventual
    /// PLAY_WORD action (emitted on TONE_COMPLETE while in RESOLVING)
    /// can carry the right word_id.
    uint32_t         last_resolved_word_id;
} spell_ui_core_t;

// =============================================================================
// API
// =============================================================================

/**
 * Reset to IDLE with no memoized word_id.
 */
void spell_ui_core_init(spell_ui_core_t *ui);

/**
 * Process one event. Returns the (next_state, action_list) the IDF wrapper
 * must materialize. Updates @p ui->state and @p ui->last_resolved_word_id
 * in place.
 *
 * Every (state, event) pair is enumerated in the implementation; cells
 * that should not fire in practice (e.g. WORD_RESOLVED in IDLE) self-loop
 * with no actions and are documented inline.
 */
spell_ui_step_t spell_ui_core_step(spell_ui_core_t *ui, spell_ui_event_t evt);

#ifdef __cplusplus
}  // extern "C"
#endif
