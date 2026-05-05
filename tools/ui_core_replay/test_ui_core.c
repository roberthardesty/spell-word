/**
 * @file test_ui_core.c
 * @brief Unit tests for the host-portable UI transition table.
 *
 * Drives `spell_ui_core_step()` with synthetic event sequences and asserts
 * the (next_state, action_list) returned. Doesn't depend on ESP-IDF or
 * any timing/IO; pure (state, event) → (state, actions).
 *
 * Build + run:
 *   cd tools/ui_core_replay && make test
 *
 * Each test function prints "PASS: <name>" on success or aborts with a
 * detailed FAIL message. Exit code 0 = all pass.
 */

#include "ui_core.h"
#include "spell_config.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Test harness primitives
// ---------------------------------------------------------------------------

#define ASSERT_EQ_INT(actual, expected, msg) do { \
    long _a = (long)(actual), _e = (long)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %ld, expected %ld\n", \
                __func__, (msg), _a, _e); \
        exit(1); \
    } \
} while (0)

#define ASSERT_EQ_STATE(actual, expected) \
    ASSERT_EQ_INT((int)(actual), (int)(expected), "state mismatch")

static const char *act_name(spell_ui_action_kind_t k)
{
    switch (k) {
    case SPELL_UI_ACT_PLAY_TONE:               return "PLAY_TONE";
    case SPELL_UI_ACT_PLAY_WORD:               return "PLAY_WORD";
    case SPELL_UI_ACT_CANCEL_PLAYBACK:         return "CANCEL_PLAYBACK";
    case SPELL_UI_ACT_SET_SEGMENTER_ACTIVE:    return "SET_SEG_ACTIVE";
    case SPELL_UI_ACT_START_ARM_TIMEOUT:       return "START_ARM_TIMEOUT";
    case SPELL_UI_ACT_CANCEL_ARM_TIMEOUT:      return "CANCEL_ARM_TIMEOUT";
    case SPELL_UI_ACT_KIND_COUNT:              return "<COUNT>";
    }
    return "<unknown>";
}

static int find_action(const spell_ui_step_t *step, spell_ui_action_kind_t kind)
{
    for (int i = 0; i < step->n_actions; i++) {
        if (step->actions[i].kind == kind) return i;
    }
    return -1;
}

static void dump_step(const char *test, const spell_ui_step_t *step)
{
    fprintf(stderr, "  step in %s: next_state=%d, n_actions=%d\n",
            test, (int)step->next_state, step->n_actions);
    for (int i = 0; i < step->n_actions; i++) {
        fprintf(stderr, "    [%d] %s\n", i, act_name(step->actions[i].kind));
    }
}

#define ASSERT_HAS_ACTION(step, kind) do { \
    if (find_action(&(step), (kind)) < 0) { \
        fprintf(stderr, "FAIL [%s]: missing action %s\n", \
                __func__, act_name(kind)); \
        dump_step(__func__, &(step)); \
        exit(1); \
    } \
} while (0)

#define ASSERT_NO_ACTIONS(step) do { \
    if ((step).n_actions != 0) { \
        fprintf(stderr, "FAIL [%s]: expected no actions, got %d\n", \
                __func__, (step).n_actions); \
        dump_step(__func__, &(step)); \
        exit(1); \
    } \
} while (0)

#define ASSERT_N_ACTIONS(step, n) do { \
    if ((step).n_actions != (n)) { \
        fprintf(stderr, "FAIL [%s]: expected %d actions, got %d\n", \
                __func__, (n), (step).n_actions); \
        dump_step(__func__, &(step)); \
        exit(1); \
    } \
} while (0)

// Convenience constructors for events.
static spell_ui_event_t evt_simple(spell_ui_event_kind_t k)
{
    spell_ui_event_t e = { .kind = k, .word_id = 0 };
    return e;
}

static spell_ui_event_t evt_word_resolved(uint32_t word_id)
{
    spell_ui_event_t e = { .kind = SPELL_UI_EVT_WORD_RESOLVED, .word_id = word_id };
    return e;
}

// ---------------------------------------------------------------------------
// Tests — one per AT, plus a few full-path scenarios
// ---------------------------------------------------------------------------

// AT: button-during-IDLE → ARMED + confirmation tone
static void test_idle_button_press_arms_with_confirmation_tone(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ASSERT_EQ_STATE(ui.state, SPELL_UI_STATE_IDLE);

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_ARMED);
    ASSERT_EQ_STATE(ui.state, SPELL_UI_STATE_ARMED);
    ASSERT_N_ACTIONS(step, 2);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_PLAY_TONE);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_START_ARM_TIMEOUT);

    int idx = find_action(&step, SPELL_UI_ACT_PLAY_TONE);
    ASSERT_EQ_INT(step.actions[idx].payload.tone_id, SPELL_TONE_CONFIRMATION,
                  "tone id should be confirmation");
    printf("PASS: %s\n", __func__);
}

// AT: button-during-ARMED ignored
static void test_armed_button_press_ignored(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));
    ASSERT_EQ_STATE(ui.state, SPELL_UI_STATE_ARMED);

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_ARMED);
    ASSERT_NO_ACTIONS(step);
    printf("PASS: %s\n", __func__);
}

// Confirmation TONE_COMPLETE while ARMED → segmenter activates
static void test_armed_tone_complete_activates_segmenter(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_ARMED);
    ASSERT_N_ACTIONS(step, 1);
    int idx = find_action(&step, SPELL_UI_ACT_SET_SEGMENTER_ACTIVE);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing SET_SEG_ACTIVE\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.seg_active, 1,
                  "should activate segmenter on tone complete");
    printf("PASS: %s\n", __func__);
}

// AT: arm-timeout from ARMED → IDLE
static void test_armed_arm_timeout_goes_idle(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    ASSERT_EQ_STATE(ui.state, SPELL_UI_STATE_ARMED);

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_ARM_TIMEOUT));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_IDLE);
    ASSERT_N_ACTIONS(step, 1);
    int idx = find_action(&step, SPELL_UI_ACT_SET_SEGMENTER_ACTIVE);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing SET_SEG_ACTIVE on disarm\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.seg_active, 0,
                  "should deactivate segmenter on arm-timeout");
    printf("PASS: %s\n", __func__);
}

// AT: first-letter-onset from ARMED → SPELLING
static void test_armed_first_letter_onset_to_spelling(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_FIRST_LETTER_ONSET));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_SPELLING);
    ASSERT_N_ACTIONS(step, 1);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_CANCEL_ARM_TIMEOUT);
    printf("PASS: %s\n", __func__);
}

// AT: EOW from SPELLING → RESOLVING (mute mic, no resolved tone yet)
static void test_spelling_eow_to_resolving(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_SPELLING;

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_END_OF_WORD));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_RESOLVING);
    ASSERT_N_ACTIONS(step, 1);
    int idx = find_action(&step, SPELL_UI_ACT_SET_SEGMENTER_ACTIVE);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing SET_SEG_ACTIVE on EOW\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.seg_active, 0,
                  "should deactivate segmenter on EOW");
    printf("PASS: %s\n", __func__);
}

// AT: early-commit-window with predicate-met → RESOLVING.
//     Surfaces here as WORD_RESOLVED arriving while still in SPELLING
//     (the decoder published before segmenter EOW).
static void test_spelling_word_resolved_early_to_resolving(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_SPELLING;

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_word_resolved(7));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_RESOLVING);
    ASSERT_N_ACTIONS(step, 2);
    int seg_idx = find_action(&step, SPELL_UI_ACT_SET_SEGMENTER_ACTIVE);
    int tone_idx = find_action(&step, SPELL_UI_ACT_PLAY_TONE);
    if (seg_idx < 0 || tone_idx < 0) {
        fprintf(stderr, "FAIL [%s]: expected SET_SEG_ACTIVE and PLAY_TONE\n", __func__);
        dump_step(__func__, &step);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[seg_idx].payload.seg_active, 0,
                  "should deactivate segmenter on early commit");
    ASSERT_EQ_INT(step.actions[tone_idx].payload.tone_id, SPELL_TONE_RESOLVED,
                  "should play resolved tone on early commit");
    ASSERT_EQ_INT(ui.last_resolved_word_id, 7, "word_id should be memoized");
    printf("PASS: %s\n", __func__);
}

// AT: resolved-tone-then-PLAYING. Full-EOW commit path:
// EOW → RESOLVING, WORD_RESOLVED → resolved tone, TONE_COMPLETE → PLAYING + PLAY_WORD.
static void test_resolving_tone_complete_plays_word(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_SPELLING;
    (void)spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_END_OF_WORD));
    ASSERT_EQ_STATE(ui.state, SPELL_UI_STATE_RESOLVING);

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_word_resolved(42));
    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_RESOLVING);
    ASSERT_N_ACTIONS(step, 1);
    int idx = find_action(&step, SPELL_UI_ACT_PLAY_TONE);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing PLAY_TONE after WORD_RESOLVED\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.tone_id, SPELL_TONE_RESOLVED,
                  "tone after WORD_RESOLVED should be resolved tone");
    ASSERT_EQ_INT(ui.last_resolved_word_id, 42, "word_id should be memoized");

    step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_PLAYING);
    ASSERT_N_ACTIONS(step, 1);
    idx = find_action(&step, SPELL_UI_ACT_PLAY_WORD);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing PLAY_WORD on tone complete\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.word_id, 42,
                  "PLAY_WORD should carry memoized word_id");
    printf("PASS: %s\n", __func__);
}

// AT: cancel + re-arm from PLAYING.
static void test_playing_button_cancels_and_rearms(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_PLAYING;
    ui.last_resolved_word_id = 99;

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_ARMED);
    ASSERT_N_ACTIONS(step, 3);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_CANCEL_PLAYBACK);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_PLAY_TONE);
    ASSERT_HAS_ACTION(step, SPELL_UI_ACT_START_ARM_TIMEOUT);

    int tone_idx = find_action(&step, SPELL_UI_ACT_PLAY_TONE);
    ASSERT_EQ_INT(step.actions[tone_idx].payload.tone_id, SPELL_TONE_CONFIRMATION,
                  "re-arm should play confirmation tone");

    // Action ordering: CANCEL_PLAYBACK must precede PLAY_TONE so the amp
    // is muted before we try to drive the new tone (PRD §"Cancel ordering
    // contract"). The IDF wrapper consumes actions in array order.
    int cancel_idx = find_action(&step, SPELL_UI_ACT_CANCEL_PLAYBACK);
    if (cancel_idx > tone_idx) {
        fprintf(stderr, "FAIL [%s]: CANCEL_PLAYBACK must precede PLAY_TONE "
                "(got cancel@%d, tone@%d)\n", __func__, cancel_idx, tone_idx);
        exit(1);
    }
    printf("PASS: %s\n", __func__);
}

// AT: abstain from RESOLVING → IDLE with error chirp.
static void test_resolving_abstain_to_idle_with_error_chirp(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_RESOLVING;

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_WORD_ABSTAIN));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_IDLE);
    ASSERT_N_ACTIONS(step, 1);
    int idx = find_action(&step, SPELL_UI_ACT_PLAY_TONE);
    if (idx < 0) {
        fprintf(stderr, "FAIL [%s]: missing PLAY_TONE on abstain\n", __func__);
        exit(1);
    }
    ASSERT_EQ_INT(step.actions[idx].payload.tone_id, SPELL_TONE_ERROR_CHIRP,
                  "abstain should play error chirp");
    printf("PASS: %s\n", __func__);
}

// PLAYING + WORD_PLAYBACK_COMPLETE → IDLE silently.
static void test_playing_word_complete_to_idle(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);
    ui.state = SPELL_UI_STATE_PLAYING;

    spell_ui_step_t step = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE));

    ASSERT_EQ_STATE(step.next_state, SPELL_UI_STATE_IDLE);
    ASSERT_NO_ACTIONS(step);
    printf("PASS: %s\n", __func__);
}

// Full happy path (full-EOW commit): IDLE → ARMED → SPELLING → RESOLVING → PLAYING → IDLE.
static void test_full_happy_path_full_eow(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);

    spell_ui_step_t s;

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_ARMED);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_ARMED);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_FIRST_LETTER_ONSET));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_SPELLING);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_END_OF_WORD));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_RESOLVING);

    s = spell_ui_core_step(&ui, evt_word_resolved(123));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_RESOLVING);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_PLAYING);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_IDLE);

    printf("PASS: %s\n", __func__);
}

// Full happy path (early commit): IDLE → ARMED → SPELLING → RESOLVING (via WORD_RESOLVED in SPELLING)
// → PLAYING → IDLE. Skips the explicit EOW step.
static void test_full_happy_path_early_commit(void)
{
    spell_ui_core_t ui;
    spell_ui_core_init(&ui);

    spell_ui_step_t s;

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_BUTTON_PRESSED));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_ARMED);
    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_FIRST_LETTER_ONSET));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_SPELLING);

    // Decoder commits early, before segmenter EOW. UI: SPELLING → RESOLVING
    // with mic muted and resolved tone kicked off in one step.
    s = spell_ui_core_step(&ui, evt_word_resolved(7));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_RESOLVING);
    ASSERT_N_ACTIONS(s, 2);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_TONE_COMPLETE));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_PLAYING);

    s = spell_ui_core_step(&ui, evt_simple(SPELL_UI_EVT_WORD_PLAYBACK_COMPLETE));
    ASSERT_EQ_STATE(s.next_state, SPELL_UI_STATE_IDLE);

    printf("PASS: %s\n", __func__);
}

// Defensive coverage: every (state, event) cell returns without crashing
// and either self-loops or makes a documented transition. Catches a future
// reader who adds a new event-kind without enumerating every existing state.
static void test_every_state_event_pair_handled(void)
{
    for (int s = 0; s <= SPELL_UI_STATE_PLAYING; s++) {
        for (int e = 0; e < SPELL_UI_EVT_KIND_COUNT; e++) {
            spell_ui_core_t ui;
            spell_ui_core_init(&ui);
            ui.state = (spell_ui_state_t)s;
            spell_ui_event_t evt = { .kind = (spell_ui_event_kind_t)e, .word_id = 0 };
            spell_ui_step_t step = spell_ui_core_step(&ui, evt);

            // next_state is one of the 5 valid states.
            if (step.next_state < SPELL_UI_STATE_IDLE ||
                step.next_state > SPELL_UI_STATE_PLAYING) {
                fprintf(stderr, "FAIL [%s]: state %d × event %d produced bad next_state %d\n",
                        __func__, s, e, (int)step.next_state);
                exit(1);
            }
            // Action count within bounds.
            if (step.n_actions < 0 || step.n_actions > SPELL_UI_MAX_ACTIONS_PER_STEP) {
                fprintf(stderr, "FAIL [%s]: state %d × event %d emitted %d actions\n",
                        __func__, s, e, step.n_actions);
                exit(1);
            }
        }
    }
    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void)
{
    test_idle_button_press_arms_with_confirmation_tone();
    test_armed_button_press_ignored();
    test_armed_tone_complete_activates_segmenter();
    test_armed_arm_timeout_goes_idle();
    test_armed_first_letter_onset_to_spelling();
    test_spelling_eow_to_resolving();
    test_spelling_word_resolved_early_to_resolving();
    test_resolving_tone_complete_plays_word();
    test_playing_button_cancels_and_rearms();
    test_resolving_abstain_to_idle_with_error_chirp();
    test_playing_word_complete_to_idle();
    test_full_happy_path_full_eow();
    test_full_happy_path_early_commit();
    test_every_state_event_pair_handled();

    printf("\nAll tests passed.\n");
    return 0;
}
