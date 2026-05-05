/**
 * @file ui_core.c
 * @brief Pure-C UI state-machine transition table — implementation.
 *
 * Every (state, event) cell is enumerated explicitly, including cells that
 * should not fire in steady-state. Defensive cells self-loop with no
 * actions; the inline comment notes why the cell shouldn't fire so the
 * row stays auditable.
 */

#include "ui_core.h"

#include <assert.h>
#include <string.h>

#include "spell_config.h"   // SPELL_TONE_*

// =============================================================================
// Internal helpers
// =============================================================================

static void emit(spell_ui_step_t *step, spell_ui_action_t act)
{
    assert(step->n_actions < SPELL_UI_MAX_ACTIONS_PER_STEP);
    step->actions[step->n_actions++] = act;
}

static spell_ui_action_t act_play_tone(uint8_t tone_id)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_PLAY_TONE };
    a.payload.tone_id = tone_id;
    return a;
}

static spell_ui_action_t act_play_word(uint32_t word_id)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_PLAY_WORD };
    a.payload.word_id = word_id;
    return a;
}

static spell_ui_action_t act_cancel_playback(void)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_CANCEL_PLAYBACK };
    return a;
}

static spell_ui_action_t act_set_seg_active(bool active)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_SET_SEGMENTER_ACTIVE };
    a.payload.seg_active = active;
    return a;
}

static spell_ui_action_t act_start_arm_timeout(void)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_START_ARM_TIMEOUT };
    return a;
}

static spell_ui_action_t act_cancel_arm_timeout(void)
{
    spell_ui_action_t a = { .kind = SPELL_UI_ACT_CANCEL_ARM_TIMEOUT };
    return a;
}

// =============================================================================
// Per-state step functions
// =============================================================================
//
// Each function reads ui->state and the incoming event, writes step->next_state
// and step->actions. Cells that cannot legitimately fire are self-loops with no
// actions; the inline note explains why.

static void step_idle(spell_ui_core_t *ui, spell_ui_event_t evt, spell_ui_step_t *step)
{
    (void)ui;
    switch (evt.kind) {
    case SPELL_UI_EVT_BUTTON_PRESSED:
        // IDLE → ARMED: play confirmation tone, start arm timer. Segmenter
        // stays inactive until TONE_COMPLETE so the device's own tone does
        // not get treated as the first letter (US-2).
        step->next_state = SPELL_UI_STATE_ARMED;
        emit(step, act_play_tone(SPELL_TONE_CONFIRMATION));
        emit(step, act_start_arm_timeout());
        break;
    case SPELL_UI_EVT_ARM_TIMEOUT:
    case SPELL_UI_EVT_FIRST_LETTER_ONSET:
    case SPELL_UI_EVT_END_OF_WORD:
    case SPELL_UI_EVT_WORD_RESOLVED:
    case SPELL_UI_EVT_WORD_ABSTAIN:
    case SPELL_UI_EVT_TONE_COMPLETE:
    case SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE:
        // No timer, no segmenter, no decoder, no playback in IDLE — these
        // events should not arrive. If a stale one slips through (e.g. a
        // late TONE_COMPLETE after a cancel), drop it silently.
        break;
    case SPELL_UI_EVT_KIND_COUNT:
        break;  // unreachable; sentinel value
    }
}

static void step_armed(spell_ui_core_t *ui, spell_ui_event_t evt, spell_ui_step_t *step)
{
    (void)ui;
    switch (evt.kind) {
    case SPELL_UI_EVT_BUTTON_PRESSED:
        // Ignored per US-1/US-2/PRD: a second press while waiting for audio
        // would either bounce or accidentally re-arm. Stay in ARMED.
        break;
    case SPELL_UI_EVT_ARM_TIMEOUT:
        // Auto-disarm: no audio in SPELL_ARM_TIMEOUT_MS. Mute mic and go IDLE.
        step->next_state = SPELL_UI_STATE_IDLE;
        emit(step, act_set_seg_active(false));
        break;
    case SPELL_UI_EVT_FIRST_LETTER_ONSET:
        // First letter — cancel the arm-timeout and move to SPELLING. The
        // segmenter is already active (turned on at TONE_COMPLETE).
        step->next_state = SPELL_UI_STATE_SPELLING;
        emit(step, act_cancel_arm_timeout());
        break;
    case SPELL_UI_EVT_TONE_COMPLETE:
        // Confirmation tone finished — now safe to start listening for
        // letters without re-triggering on our own audio (US-2).
        emit(step, act_set_seg_active(true));
        break;
    case SPELL_UI_EVT_END_OF_WORD:
    case SPELL_UI_EVT_WORD_RESOLVED:
    case SPELL_UI_EVT_WORD_ABSTAIN:
    case SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE:
        // Decoder + playback events are impossible in ARMED (no spelling
        // session in flight, no playback running). Stale events silently
        // dropped.
        break;
    case SPELL_UI_EVT_KIND_COUNT:
        break;
    }
}

static void step_spelling(spell_ui_core_t *ui, spell_ui_event_t evt, spell_ui_step_t *step)
{
    switch (evt.kind) {
    case SPELL_UI_EVT_BUTTON_PRESSED:
        // Ignored — user is mid-spelling. Don't let an accidental press
        // abort a word that's being typed letter-by-letter.
        break;
    case SPELL_UI_EVT_ARM_TIMEOUT:
        // Timer was cancelled at FIRST_LETTER_ONSET. A late fire is a stale
        // event (race between cancel and timer task) — drop it.
        break;
    case SPELL_UI_EVT_FIRST_LETTER_ONSET:
        // The "first" event by definition only triggers ARMED → SPELLING.
        // Subsequent letter onsets are no-ops at the UI layer.
        break;
    case SPELL_UI_EVT_END_OF_WORD:
        // Segmenter declared end-of-word. Mute mic, wait for decoder to
        // publish WORD_RESOLVED or WORD_ABSTAIN.
        step->next_state = SPELL_UI_STATE_RESOLVING;
        emit(step, act_set_seg_active(false));
        break;
    case SPELL_UI_EVT_WORD_RESOLVED:
        // Aggressive early-commit path (PRD §"Aggressive early-commit
        // predicate"): decoder published before segmenter EOW. Move to
        // RESOLVING, mute mic, kick off the resolved tone immediately.
        ui->last_resolved_word_id = evt.word_id;
        step->next_state = SPELL_UI_STATE_RESOLVING;
        emit(step, act_set_seg_active(false));
        emit(step, act_play_tone(SPELL_TONE_RESOLVED));
        break;
    case SPELL_UI_EVT_WORD_ABSTAIN:
        // Per PRD §"Aggressive early-commit predicate", abstention only
        // fires on full EOW — so this path is unreachable in steady-state.
        // If a future decoder change emits abstain mid-spelling, fail safe:
        // mute mic, error chirp, back to IDLE (mirrors RESOLVING + abstain).
        step->next_state = SPELL_UI_STATE_IDLE;
        emit(step, act_set_seg_active(false));
        emit(step, act_play_tone(SPELL_TONE_ERROR_CHIRP));
        break;
    case SPELL_UI_EVT_TONE_COMPLETE:
        // Confirmation tone may complete after FIRST_LETTER_ONSET in fast
        // spellers; nothing for us to do — segmenter was already activated
        // when the tone completed in ARMED, but if FIRST_LETTER_ONSET fired
        // first that means the user was already vocalizing while the tone
        // was playing. Either way, no UI-layer action.
        break;
    case SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE:
        // No word is playing. Stale.
        break;
    case SPELL_UI_EVT_KIND_COUNT:
        break;
    }
}

static void step_resolving(spell_ui_core_t *ui, spell_ui_event_t evt, spell_ui_step_t *step)
{
    switch (evt.kind) {
    case SPELL_UI_EVT_BUTTON_PRESSED:
        // Resolving is a brief window (≤ ~80 ms before resolved tone fires).
        // A press here is more likely a mis-press than an intentional
        // cancel; the cancel-and-re-arm flow is for PLAYING, not RESOLVING.
        break;
    case SPELL_UI_EVT_ARM_TIMEOUT:
        // Timer is cancelled by now.
        break;
    case SPELL_UI_EVT_FIRST_LETTER_ONSET:
    case SPELL_UI_EVT_END_OF_WORD:
        // Segmenter was muted on entry to RESOLVING.
        break;
    case SPELL_UI_EVT_WORD_RESOLVED:
        // Full-EOW commit path: decoder publishes after segmenter EOW.
        // (Early-commit path lands in SPELLING + WORD_RESOLVED instead.)
        // Memoize word_id; play the resolved tone — PLAYING follows on
        // TONE_COMPLETE.
        ui->last_resolved_word_id = evt.word_id;
        emit(step, act_play_tone(SPELL_TONE_RESOLVED));
        break;
    case SPELL_UI_EVT_WORD_ABSTAIN:
        // Decoder declined to commit (margin too tight or overflow).
        // Error chirp, back to IDLE.
        step->next_state = SPELL_UI_STATE_IDLE;
        emit(step, act_play_tone(SPELL_TONE_ERROR_CHIRP));
        break;
    case SPELL_UI_EVT_TONE_COMPLETE:
        // Resolved tone finished — now play the word. PLAY_WORD carries the
        // memoized word_id.
        step->next_state = SPELL_UI_STATE_PLAYING;
        emit(step, act_play_word(ui->last_resolved_word_id));
        break;
    case SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE:
        // No word playing yet (still on resolved tone).
        break;
    case SPELL_UI_EVT_KIND_COUNT:
        break;
    }
}

static void step_playing(spell_ui_core_t *ui, spell_ui_event_t evt, spell_ui_step_t *step)
{
    (void)ui;
    switch (evt.kind) {
    case SPELL_UI_EVT_BUTTON_PRESSED:
        // Cancel-and-re-arm (US-6, US-11): stop playback and restart the
        // confirmation-tone-then-arm sequence in one step. The IDF wrapper
        // is responsible for the cancel ordering contract (PRD §"Cancel
        // ordering contract") — amp-low → cancel-flag → ack → I2S TX off.
        step->next_state = SPELL_UI_STATE_ARMED;
        emit(step, act_cancel_playback());
        emit(step, act_play_tone(SPELL_TONE_CONFIRMATION));
        emit(step, act_start_arm_timeout());
        break;
    case SPELL_UI_EVT_ARM_TIMEOUT:
    case SPELL_UI_EVT_FIRST_LETTER_ONSET:
    case SPELL_UI_EVT_END_OF_WORD:
    case SPELL_UI_EVT_WORD_RESOLVED:
    case SPELL_UI_EVT_WORD_ABSTAIN:
        // No timer, no segmenter, no decoder running while a word plays.
        break;
    case SPELL_UI_EVT_TONE_COMPLETE:
        // The resolved tone's TONE_COMPLETE fires in RESOLVING and triggers
        // the PLAY_WORD that put us here. A TONE_COMPLETE arriving in
        // PLAYING is therefore stale.
        break;
    case SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE:
        // Word finished naturally. Back to IDLE.
        step->next_state = SPELL_UI_STATE_IDLE;
        break;
    case SPELL_UI_EVT_KIND_COUNT:
        break;
    }
}

// =============================================================================
// Public API
// =============================================================================

void spell_ui_core_init(spell_ui_core_t *ui)
{
    memset(ui, 0, sizeof(*ui));
    ui->state = SPELL_UI_STATE_IDLE;
}

spell_ui_step_t spell_ui_core_step(spell_ui_core_t *ui, spell_ui_event_t evt)
{
    spell_ui_step_t step;
    memset(&step, 0, sizeof(step));
    step.next_state = ui->state;   // self-loop default; per-cell code may override

    switch (ui->state) {
    case SPELL_UI_STATE_IDLE:      step_idle(ui, evt, &step);      break;
    case SPELL_UI_STATE_ARMED:     step_armed(ui, evt, &step);     break;
    case SPELL_UI_STATE_SPELLING:  step_spelling(ui, evt, &step);  break;
    case SPELL_UI_STATE_RESOLVING: step_resolving(ui, evt, &step); break;
    case SPELL_UI_STATE_PLAYING:   step_playing(ui, evt, &step);   break;
    }

    ui->state = step.next_state;
    return step;
}
