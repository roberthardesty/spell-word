/**
 * @file main.c
 * @brief CLI driver — reads a WAV, runs the VAD, prints the segmentation.
 *
 * Lets you tune VAD parameters against recorded audio without re-flashing.
 * The same segmenter_core.c that runs on-device is linked here, so the
 * algorithm is bit-identical (modulo float-precision differences between
 * Xtensa and host).
 *
 * Usage:
 *   ./segmenter_replay [options] input.wav
 *
 * Options (all override the SPELL_VAD_* defaults from spell_config.h):
 *   --on DBFS            Onset threshold (default -38.0)
 *   --off DBFS           Offset threshold (default -42.0)
 *   --ema N              EMA alpha × 100 (default 35; smaller = smoother)
 *   --off-frames N       Sub-off frames to confirm offset (default 5)
 *   --min-letter MS      Reject letters shorter than this (default 150)
 *   --max-letter MS      Force-split letters longer than this (default 800)
 *   --eow MS             End-of-word silence (default 1200)
 *   --preroll MS         Pre-roll context (default 200)
 *   --dump-letters DIR   Write each emitted utterance as DIR/letter_NNN.wav
 *   --quiet              Suppress per-frame logging; just print summary
 *   --help               Show usage
 */

#include "segmenter_core.h"
#include "wav.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Defaults — keep in lockstep with components/spell_common/include/spell_config.h.
#define DEF_ON_DBFS         -38.0f
#define DEF_OFF_DBFS        -42.0f
#define DEF_EMA_ALPHA_X100  35
#define DEF_OFF_FRAMES      5
#define DEF_MIN_LETTER_MS   150
#define DEF_MAX_LETTER_MS   800
#define DEF_EOW_MS          1200
#define DEF_PREROLL_MS      200

#define FRAME_SAMPLES       256
#define SAMPLE_RATE         16000

static void print_usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [options] input.wav\n"
        "\n"
        "options:\n"
        "  --on DBFS            onset threshold (default %.1f)\n"
        "  --off DBFS           offset threshold (default %.1f)\n"
        "  --ema N              EMA alpha × 100 (default %d)\n"
        "  --off-frames N       sub-off frames to confirm (default %d)\n"
        "  --min-letter MS      reject below this duration (default %d)\n"
        "  --max-letter MS      force-split above this (default %d)\n"
        "  --eow MS             end-of-word silence (default %d)\n"
        "  --preroll MS         pre-roll context (default %d)\n"
        "  --dump-letters DIR   write each utterance as DIR/letter_NNN.wav\n"
        "  --quiet              suppress per-event log\n"
        "  --help               this message\n",
        argv0,
        DEF_ON_DBFS, DEF_OFF_DBFS, DEF_EMA_ALPHA_X100, DEF_OFF_FRAMES,
        DEF_MIN_LETTER_MS, DEF_MAX_LETTER_MS, DEF_EOW_MS, DEF_PREROLL_MS);
}

static int ms_to_frames(int ms) {
    return (ms * SAMPLE_RATE) / (1000 * FRAME_SAMPLES);
}

int main(int argc, char **argv)
{
    float on_dbfs        = DEF_ON_DBFS;
    float off_dbfs       = DEF_OFF_DBFS;
    int   ema_alpha_x100 = DEF_EMA_ALPHA_X100;
    int   off_frames     = DEF_OFF_FRAMES;
    int   min_letter_ms  = DEF_MIN_LETTER_MS;
    int   max_letter_ms  = DEF_MAX_LETTER_MS;
    int   eow_ms         = DEF_EOW_MS;
    int   preroll_ms     = DEF_PREROLL_MS;
    const char *dump_dir = NULL;
    int   quiet          = 0;
    const char *input_wav = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--on")           && i+1<argc) on_dbfs        = (float)atof(argv[++i]);
        else if (!strcmp(a, "--off")          && i+1<argc) off_dbfs       = (float)atof(argv[++i]);
        else if (!strcmp(a, "--ema")          && i+1<argc) ema_alpha_x100 = atoi(argv[++i]);
        else if (!strcmp(a, "--off-frames")   && i+1<argc) off_frames     = atoi(argv[++i]);
        else if (!strcmp(a, "--min-letter")   && i+1<argc) min_letter_ms  = atoi(argv[++i]);
        else if (!strcmp(a, "--max-letter")   && i+1<argc) max_letter_ms  = atoi(argv[++i]);
        else if (!strcmp(a, "--eow")          && i+1<argc) eow_ms         = atoi(argv[++i]);
        else if (!strcmp(a, "--preroll")      && i+1<argc) preroll_ms     = atoi(argv[++i]);
        else if (!strcmp(a, "--dump-letters") && i+1<argc) dump_dir       = argv[++i];
        else if (!strcmp(a, "--quiet"))                    quiet          = 1;
        else if (!strcmp(a, "--help"))                   { print_usage(argv[0]); return 0; }
        else if (a[0] != '-' && !input_wav)               input_wav = a;
        else { fprintf(stderr, "unknown arg: %s\n", a); print_usage(argv[0]); return 2; }
    }
    if (!input_wav) { print_usage(argv[0]); return 2; }

    int16_t *pcm = NULL;
    size_t   n   = 0;
    int      rate;
    if (wav_read_16k_mono(input_wav, &pcm, &n, &rate) != 0) return 1;

    int preroll_samples = (preroll_ms * SAMPLE_RATE) / 1000;
    int accum_capacity  = preroll_samples
                        + (max_letter_ms * SAMPLE_RATE) / 1000
                        + off_frames * FRAME_SAMPLES + 256;

    int16_t *preroll = calloc(preroll_samples, sizeof(int16_t));
    int16_t *accum   = calloc(accum_capacity,  sizeof(int16_t));
    if (!preroll || !accum) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    seg_config_t cfg = {
        .frame_samples       = FRAME_SAMPLES,
        .sample_rate         = SAMPLE_RATE,
        .on_dbfs             = on_dbfs,
        .off_dbfs            = off_dbfs,
        .ema_alpha_x100      = ema_alpha_x100,
        .off_frames          = off_frames,
        .min_letter_frames   = ms_to_frames(min_letter_ms),
        .max_letter_frames   = ms_to_frames(max_letter_ms),
        .eow_frames          = ms_to_frames(eow_ms),
        .preroll             = preroll,
        .preroll_samples     = preroll_samples,
        .accum               = accum,
        .accum_capacity      = accum_capacity,
    };

    seg_state_t st;
    if (seg_init(&st, &cfg) != 0) {
        fprintf(stderr, "seg_init() rejected configuration — see "
                        "segmenter_core.h for validation rules\n");
        return 1;
    }

    printf("file:  %s (%.3f s, %zu samples)\n",
           input_wav, (double)n / SAMPLE_RATE, n);
    printf("vad:   on=%.1f off=%.1f ema=%d off_frames=%d "
           "min_letter=%dms max_letter=%dms eow=%dms preroll=%dms\n",
           on_dbfs, off_dbfs, ema_alpha_x100, off_frames,
           min_letter_ms, max_letter_ms, eow_ms, preroll_ms);

    int letter_idx = 0;
    int total_frames = (int)(n / FRAME_SAMPLES);

    for (int f = 0; f < total_frames; f++) {
        seg_event_t evt = seg_process_frame(&st,
                                            &pcm[f * FRAME_SAMPLES],
                                            FRAME_SAMPLES);
        int t_ms = (f + 1) * FRAME_SAMPLES * 1000 / SAMPLE_RATE;

        switch (evt.kind) {
        case SEG_EVT_NONE:
            break;
        case SEG_EVT_LETTER_EMITTED:
            letter_idx++;
            if (!quiet) {
                printf("[%6d ms] letter #%d: %d samples (%d ms), peak %.1f dBFS\n",
                       t_ms, letter_idx, evt.n_samples, evt.duration_ms,
                       evt.peak_dbfs);
            }
            if (dump_dir) {
                char path[512];
                snprintf(path, sizeof(path), "%s/letter_%03d.wav",
                         dump_dir, letter_idx);
                wav_write_16k_mono(path, evt.pcm, evt.n_samples);
            }
            break;
        case SEG_EVT_FORCE_SPLIT:
            letter_idx++;
            if (!quiet) {
                printf("[%6d ms] FORCE-SPLIT #%d: %d samples (%d ms), peak %.1f dBFS\n",
                       t_ms, letter_idx, evt.n_samples, evt.duration_ms,
                       evt.peak_dbfs);
            }
            if (dump_dir) {
                char path[512];
                snprintf(path, sizeof(path), "%s/letter_%03d_split.wav",
                         dump_dir, letter_idx);
                wav_write_16k_mono(path, evt.pcm, evt.n_samples);
            }
            break;
        case SEG_EVT_LETTER_REJECTED_SHORT:
            if (!quiet) {
                printf("[%6d ms] rejected (short blip): %d ms, peak %.1f dBFS\n",
                       t_ms, evt.duration_ms, evt.peak_dbfs);
            }
            break;
        case SEG_EVT_END_OF_WORD:
            if (!quiet) {
                printf("[%6d ms] END-OF-WORD (letters_so_far=%u)\n",
                       t_ms, evt.letters_in_word);
            }
            break;
        }
    }

    printf("\nsummary: %u emitted, %u rejected, %u force-splits, %u words\n",
           st.total_letters_emitted,
           st.total_letters_rejected_short,
           st.total_force_splits,
           st.total_words_completed);

    free(pcm);
    free(preroll);
    free(accum);
    return 0;
}
