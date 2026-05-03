/**
 * @file test_segmenter_core.c
 * @brief Unit tests for the host-portable VAD core.
 *
 * Synthesizes PCM with controlled energy envelopes and asserts that
 * seg_process_frame() emits the expected events. Doesn't depend on real
 * recordings; doesn't require ESP-IDF.
 *
 * Build + run:
 *   cd tools/segmenter_replay && make test
 *
 * Each test function prints "PASS: <name>" on success or aborts with a
 * detailed FAIL message. Exit code 0 = all pass.
 */

#include "segmenter_core.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Some libc/std combos don't expose M_PI under -std=c99. Define locally.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Test harness primitives
// ---------------------------------------------------------------------------

#define FRAME_SAMPLES   256
#define SAMPLE_RATE     16000
#define FRAME_MS        (FRAME_SAMPLES * 1000 / SAMPLE_RATE)   // 16

#define ASSERT_EQ_INT(actual, expected, msg) do { \
    int _a = (int)(actual), _e = (int)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %d, expected %d\n", \
                __func__, (msg), _a, _e); \
        exit(1); \
    } \
} while (0)

#define ASSERT_GE_INT(actual, expected, msg) do { \
    int _a = (int)(actual), _e = (int)(expected); \
    if (_a < _e) { \
        fprintf(stderr, "FAIL [%s]: %s\n  got %d, expected ≥ %d\n", \
                __func__, (msg), _a, _e); \
        exit(1); \
    } \
} while (0)

// Counters incremented from a callback-style event observer, so each test
// can declare what it expects without fishing through a returned-event log.
typedef struct {
    int  letters_emitted;
    int  letters_rejected;
    int  force_splits;
    int  end_of_words;
    int  last_letter_duration_ms;
    float last_letter_peak_dbfs;
} obs_t;

static void obs_observe(obs_t *o, const seg_event_t *evt)
{
    switch (evt->kind) {
    case SEG_EVT_NONE: break;
    case SEG_EVT_LETTER_EMITTED:
        o->letters_emitted++;
        o->last_letter_duration_ms = evt->duration_ms;
        o->last_letter_peak_dbfs   = evt->peak_dbfs;
        break;
    case SEG_EVT_LETTER_REJECTED_SHORT:
        o->letters_rejected++;
        break;
    case SEG_EVT_FORCE_SPLIT:
        o->force_splits++;
        break;
    case SEG_EVT_END_OF_WORD:
        o->end_of_words++;
        break;
    }
}

// ---------------------------------------------------------------------------
// Default config — same numbers as spell_config.h. Each test starts with
// these and may override one or two fields in place.
// ---------------------------------------------------------------------------

typedef struct {
    seg_config_t cfg;
    int16_t     *preroll;
    int16_t     *accum;
    seg_state_t  state;
} fixture_t;

static int ms_to_frames(int ms) {
    return (ms * SAMPLE_RATE) / (1000 * FRAME_SAMPLES);
}

static void fixture_init(fixture_t *fx)
{
    int preroll_samples = (200 * SAMPLE_RATE) / 1000;
    int accum_capacity  = preroll_samples
                        + (800 * SAMPLE_RATE) / 1000
                        + 5 * FRAME_SAMPLES + 256;

    fx->preroll = calloc(preroll_samples, sizeof(int16_t));
    fx->accum   = calloc(accum_capacity,  sizeof(int16_t));
    if (!fx->preroll || !fx->accum) { fprintf(stderr, "alloc failed\n"); exit(1); }

    fx->cfg = (seg_config_t){
        .frame_samples       = FRAME_SAMPLES,
        .sample_rate         = SAMPLE_RATE,
        .on_dbfs             = -38.0f,
        .off_dbfs            = -42.0f,
        .ema_alpha_x100      = 35,
        .off_frames          = 5,
        .min_letter_frames   = ms_to_frames(150),
        .max_letter_frames   = ms_to_frames(800),
        .eow_frames          = ms_to_frames(1200),
        .preroll             = fx->preroll,
        .preroll_samples     = preroll_samples,
        .accum               = fx->accum,
        .accum_capacity      = accum_capacity,
    };
}

static void fixture_apply(fixture_t *fx)
{
    int rc = seg_init(&fx->state, &fx->cfg);
    ASSERT_EQ_INT(rc, 0, "seg_init returned nonzero");
}

static void fixture_destroy(fixture_t *fx)
{
    free(fx->preroll); fx->preroll = NULL;
    free(fx->accum);   fx->accum   = NULL;
}

// ---------------------------------------------------------------------------
// PCM synthesis helpers
// ---------------------------------------------------------------------------

/// Fill a frame with zeros (≈ -inf dBFS).
static void fill_silent(int16_t *frame, int n) {
    memset(frame, 0, n * sizeof(int16_t));
}

/// Fill a frame with a 1 kHz sine of the given peak amplitude.
/// peak_amp ∈ [0, 32767]. Phase is reset every call (fine for VAD; the RMS
/// over a frame is independent of phase).
static void fill_sine(int16_t *frame, int n, float peak_amp)
{
    static double phase = 0.0;
    double dphase = 2.0 * M_PI * 1000.0 / SAMPLE_RATE;
    for (int i = 0; i < n; i++) {
        frame[i] = (int16_t)(peak_amp * sin(phase));
        phase += dphase;
        if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
    }
}

/// Drive @p ms milliseconds of @p frame_filler through the segmenter.
/// frame_filler is called with the frame buffer + sample count.
typedef void (*filler_t)(int16_t *frame, int n, void *ctx);

static void run_ms(seg_state_t *st, obs_t *obs, int ms,
                   filler_t fill, void *ctx)
{
    int n_frames = (ms * SAMPLE_RATE) / (1000 * FRAME_SAMPLES);
    int16_t frame[FRAME_SAMPLES];
    for (int i = 0; i < n_frames; i++) {
        fill(frame, FRAME_SAMPLES, ctx);
        seg_event_t evt = seg_process_frame(st, frame, FRAME_SAMPLES);
        obs_observe(obs, &evt);
    }
}

// Filler implementations (pass through filler_t pointer).
static void fill_silent_ctx(int16_t *frame, int n, void *ctx) {
    (void)ctx; fill_silent(frame, n);
}
static void fill_sine_loud_ctx(int16_t *frame, int n, void *ctx) {
    (void)ctx;
    // peak_amp 8000 → -12 dBFS RMS-ish. Comfortably above the -38 onset.
    fill_sine(frame, n, 8000.0f);
}
static void fill_sine_soft_ctx(int16_t *frame, int n, void *ctx) {
    (void)ctx;
    // peak_amp 100 → roughly -50 dBFS RMS — below the -38 onset.
    fill_sine(frame, n, 100.0f);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_silence_emits_nothing(void)
{
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, /*ms=*/3000, fill_silent_ctx, NULL);

    ASSERT_EQ_INT(obs.letters_emitted,  0, "silence emitted a letter");
    ASSERT_EQ_INT(obs.letters_rejected, 0, "silence triggered short reject");
    ASSERT_EQ_INT(obs.force_splits,     0, "silence force-split");
    ASSERT_EQ_INT(obs.end_of_words,     0, "silence emitted EOW (no word "
                                            "in flight)");
    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_single_clean_letter(void)
{
    // 200 ms loud + 600 ms silent → one letter, then EOW (silent total is
    // shorter than 1200 ms so no EOW yet) — adjust silent tail to 1300 ms.
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, 100,  fill_silent_ctx,    NULL);  // pre-roll fill
    run_ms(&fx.state, &obs, 300,  fill_sine_loud_ctx, NULL);  // letter
    run_ms(&fx.state, &obs, 1500, fill_silent_ctx,    NULL);  // post + EOW

    ASSERT_EQ_INT(obs.letters_emitted,  1, "expected exactly 1 letter");
    ASSERT_EQ_INT(obs.letters_rejected, 0, "unexpected short-reject");
    ASSERT_EQ_INT(obs.force_splits,     0, "unexpected force-split");
    ASSERT_EQ_INT(obs.end_of_words,     1, "expected exactly 1 EOW");

    // Letter duration includes the 5-frame offset confirmation tail. With
    // off_frames=5 (80 ms) and a clean 300 ms burst, the reported
    // duration_ms is ~300 + 80 = ~380 ms, ± a couple of frame quanta from
    // EMA hysteresis. Allow a generous range.
    ASSERT_GE_INT(obs.last_letter_duration_ms, 300, "duration too short");
    ASSERT_GE_INT(450, obs.last_letter_duration_ms, "duration too long");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_three_letters_one_eow(void)
{
    // Three 250 ms bursts separated by 400 ms gaps (well below EOW=1200 ms,
    // well above off-confirmation), then 1500 ms silence to trigger EOW.
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    for (int i = 0; i < 3; i++) {
        run_ms(&fx.state, &obs, 250, fill_sine_loud_ctx, NULL);
        run_ms(&fx.state, &obs, 400, fill_silent_ctx,    NULL);
    }
    run_ms(&fx.state, &obs, 1500, fill_silent_ctx, NULL);

    ASSERT_EQ_INT(obs.letters_emitted,  3, "expected 3 letters");
    ASSERT_EQ_INT(obs.end_of_words,     1, "expected exactly 1 EOW after "
                                            "the trailing silence");
    ASSERT_EQ_INT(obs.force_splits,     0, "unexpected force-split");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_short_blip_rejected(void)
{
    // 50 ms loud burst — well below the 150 ms min-letter — should be
    // counted as a short-reject, not as an emitted letter, and should NOT
    // arm word_in_flight (so no EOW fires later).
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, 100,  fill_silent_ctx,    NULL);
    run_ms(&fx.state, &obs, 50,   fill_sine_loud_ctx, NULL);
    run_ms(&fx.state, &obs, 1500, fill_silent_ctx,    NULL);

    ASSERT_EQ_INT(obs.letters_emitted,  0, "short blip emitted as a letter");
    ASSERT_EQ_INT(obs.letters_rejected, 1, "expected exactly one short reject");
    ASSERT_EQ_INT(obs.end_of_words,     0, "rejected blip should not arm "
                                            "word_in_flight");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_max_letter_force_split(void)
{
    // 1500 ms continuous loud audio — well over max-letter (800 ms).
    // Expect at least one FORCE_SPLIT.
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, 100,  fill_silent_ctx,    NULL);
    run_ms(&fx.state, &obs, 1500, fill_sine_loud_ctx, NULL);
    run_ms(&fx.state, &obs, 1500, fill_silent_ctx,    NULL);

    ASSERT_GE_INT(obs.force_splits, 1, "expected at least one force-split");
    // After force-split, the residual segment also gets emitted on the
    // trailing silence, so total letters_emitted >= force_splits + 1
    // (the final emission is counted as LETTER_EMITTED, not FORCE_SPLIT).
    ASSERT_GE_INT(obs.letters_emitted, 1, "expected residual letter "
                                          "after force-split");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_below_onset_no_emit(void)
{
    // 2 seconds of audio at -50 dBFS RMS — below the -38 onset. Nothing
    // should fire.
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, 2000, fill_sine_soft_ctx, NULL);

    ASSERT_EQ_INT(obs.letters_emitted,  0, "below-onset audio emitted a letter");
    ASSERT_EQ_INT(obs.letters_rejected, 0, "below-onset audio triggered reject");
    ASSERT_EQ_INT(obs.force_splits,     0, "below-onset audio force-split");
    ASSERT_EQ_INT(obs.end_of_words,     0, "below-onset audio fired EOW");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_eow_only_after_emission(void)
{
    // Two seconds of silence + a short blip + more silence: the blip is
    // rejected, so word_in_flight stays false, and NO EOW should fire even
    // after the long silence.
    fixture_t fx; fixture_init(&fx); fixture_apply(&fx);
    obs_t obs = {0};

    run_ms(&fx.state, &obs, 2000, fill_silent_ctx,    NULL);
    run_ms(&fx.state, &obs, 50,   fill_sine_loud_ctx, NULL);
    run_ms(&fx.state, &obs, 2000, fill_silent_ctx,    NULL);

    ASSERT_EQ_INT(obs.letters_emitted, 0, "blip should not have emitted");
    ASSERT_EQ_INT(obs.end_of_words,    0, "EOW fired without any emitted "
                                          "letter — word_in_flight bug");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

static void test_invalid_config_rejected(void)
{
    fixture_t fx; fixture_init(&fx);

    // Hysteresis has zero margin → must reject.
    fx.cfg.on_dbfs  = -42.0f;
    fx.cfg.off_dbfs = -42.0f;
    int rc = seg_init(&fx.state, &fx.cfg);
    ASSERT_GE_INT(0, rc + 1 /*want rc<0, i.e. 0 >= rc+1 iff rc <= -1*/,
                  "seg_init should reject zero-margin hysteresis");

    // Restore + try max <= min.
    fx.cfg.on_dbfs           = -38.0f;
    fx.cfg.off_dbfs          = -42.0f;
    fx.cfg.min_letter_frames = 50;
    fx.cfg.max_letter_frames = 50;
    rc = seg_init(&fx.state, &fx.cfg);
    ASSERT_GE_INT(0, rc + 1, "seg_init should reject max <= min");

    fixture_destroy(&fx);
    printf("PASS: %s\n", __func__);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void)
{
    test_invalid_config_rejected();
    test_silence_emits_nothing();
    test_below_onset_no_emit();
    test_single_clean_letter();
    test_three_letters_one_eow();
    test_short_blip_rejected();
    test_eow_only_after_emission();
    test_max_letter_force_split();

    printf("\nAll tests passed.\n");
    return 0;
}
