/**
 * @file feat_extract.c
 * @brief MFCC + Δ + ΔΔ feature extractor — pure C, esp-dsp SIMD FFT.
 *
 * Three-pass output construction over the same 3-channel tensor:
 *
 *   Pass 1: per frame, compute MFCC[k] from windowed PCM via FFT → mel →
 *           ln → DCT-II, write to channel 0.
 *   Pass 2: per frame, compute Δ[k] from neighboring channel-0 values
 *           (edge-replicated boundaries), write to channel 1.
 *   Pass 3: per frame, compute ΔΔ[k] from neighboring channel-1 values,
 *           write to channel 2.
 *
 * Pre-emphasis (y[n] = x[n] - α·x[n-1]) is applied on-the-fly during
 * frame-loading, since the operation is local and pre-applying to the
 * whole signal vs. computing it per-sample produces identical results.
 *
 * Workspace allocation strategy (carries over from log_mel.c):
 *   - FFT scratch buffer  → INTERNAL SRAM (random butterfly accesses)
 *   - Hann window         → INTERNAL SRAM (hot inner loop)
 *   - Sparse mel bounds   → INTERNAL SRAM (tiny, hot)
 *   - DCT matrix          → INTERNAL SRAM (3.2 KB, hot)
 *   - Mel filterbank      → PSRAM (40 KB, sequential access tolerates SPI)
 */

#include "feat_extract.h"
#include "spell_config.h"

#include <math.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "dsps_fft2r.h"

static const char *TAG = "feat_extract";

// Magnitude bins kept (DC through Nyquist).
#define N_FFT_BINS     (SPELL_MEL_N_FFT / 2 + 1)         // 257

// ── Workspace ─────────────────────────────────────────────────────────

static float *s_hann_window    = NULL;  // [SPELL_MEL_WIN_LENGTH]
static float *s_mel_filterbank = NULL;  // [N_MELS × N_FFT_BINS]
static float *s_fft_buf        = NULL;  // [SPELL_MEL_N_FFT * 2]
static float *s_dct_matrix     = NULL;  // [SPELL_MFCC_N_COEFFS × SPELL_MEL_N_MELS]

typedef struct { int lo; int hi; } mel_bound_t;
static mel_bound_t *s_mel_bounds = NULL; // [SPELL_MEL_N_MELS]

// ── Mel-scale helpers (HTK formulation) ───────────────────────────────
//
// NOTE — librosa's mfcc() defaults to htk=False (Slaney piecewise scale).
// If the training pipeline uses librosa with default htk, this firmware
// will not match exactly. Confirm against training code; if needed, swap
// hz_to_mel / mel_to_hz for the Slaney variant.
//
static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void build_mel_filterbank(void)
{
    const float fft_bin_hz = (float)SPELL_SAMPLE_RATE / SPELL_MEL_N_FFT;

    float mel_min = hz_to_mel(SPELL_MEL_FMIN);
    float mel_max = hz_to_mel(SPELL_MEL_FMAX);

    float mel_points[SPELL_MEL_N_MELS + 2];
    for (int i = 0; i < SPELL_MEL_N_MELS + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (SPELL_MEL_N_MELS + 1);
        mel_points[i] = mel_to_hz(mel) / fft_bin_hz;
    }

    memset(s_mel_filterbank, 0,
           SPELL_MEL_N_MELS * N_FFT_BINS * sizeof(float));

    for (int m = 0; m < SPELL_MEL_N_MELS; m++) {
        float left   = mel_points[m];
        float center = mel_points[m + 1];
        float right  = mel_points[m + 2];

        int lo = N_FFT_BINS, hi = 0;
        for (int k = 0; k < N_FFT_BINS; k++) {
            float fk = (float)k;
            if (fk >= left && fk <= center) {
                s_mel_filterbank[m * N_FFT_BINS + k] =
                    (fk - left) / (center - left);
                if (k < lo) lo = k;
                if (k + 1 > hi) hi = k + 1;
            } else if (fk > center && fk <= right) {
                s_mel_filterbank[m * N_FFT_BINS + k] =
                    (right - fk) / (right - center);
                if (k < lo) lo = k;
                if (k + 1 > hi) hi = k + 1;
            }
        }
        s_mel_bounds[m].lo = lo;
        s_mel_bounds[m].hi = hi;
    }
}

// ── DCT-II orthonormal matrix ─────────────────────────────────────────
//
// MFCC[k] = sum_{m=0..M-1} log_mel[m] * D[k][m]
//
// where, with M = SPELL_MEL_N_MELS:
//   D[0][m] = sqrt(1/M)
//   D[k][m] = sqrt(2/M) * cos(π * k * (m + 0.5) / M)   for k > 0
//
// This is the orthonormal DCT-II that scipy.fftpack.dct(norm='ortho') and
// librosa.feature.mfcc both use.
//
static void build_dct_matrix(void)
{
    const int M = SPELL_MEL_N_MELS;
    const float scale_zero = sqrtf(1.0f / (float)M);
    const float scale_rest = sqrtf(2.0f / (float)M);

    for (int k = 0; k < SPELL_MFCC_N_COEFFS; k++) {
        for (int m = 0; m < M; m++) {
            float c;
            if (k == 0) {
                c = scale_zero;
            } else {
                c = scale_rest * cosf((float)M_PI * k *
                                      ((float)m + 0.5f) / (float)M);
            }
            s_dct_matrix[k * M + m] = c;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────

esp_err_t feat_extract_init(void)
{
    s_hann_window = heap_caps_malloc(
        SPELL_MEL_WIN_LENGTH * sizeof(float),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    s_mel_filterbank = heap_caps_malloc(
        SPELL_MEL_N_MELS * N_FFT_BINS * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    s_mel_bounds = heap_caps_malloc(
        SPELL_MEL_N_MELS * sizeof(mel_bound_t),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    s_dct_matrix = heap_caps_malloc(
        SPELL_MFCC_N_COEFFS * SPELL_MEL_N_MELS * sizeof(float),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    s_fft_buf = heap_caps_malloc(
        SPELL_MEL_N_FFT * 2 * sizeof(float),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (!s_hann_window || !s_mel_filterbank || !s_mel_bounds ||
        !s_dct_matrix  || !s_fft_buf) {
        ESP_LOGE(TAG, "alloc failed for feature workspace");
        feat_extract_deinit();
        return ESP_ERR_NO_MEM;
    }

    for (int i = 0; i < SPELL_MEL_WIN_LENGTH; i++) {
        s_hann_window[i] = 0.5f * (1.0f - cosf(
            2.0f * (float)M_PI * i / (SPELL_MEL_WIN_LENGTH - 1)));
    }

    build_mel_filterbank();
    build_dct_matrix();

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, SPELL_MEL_N_FFT);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %s", esp_err_to_name(ret));
        feat_extract_deinit();
        return ret;
    }

    ESP_LOGI(TAG, "feat init: %d-pt FFT, %d mels, %d MFCC, %d frames, %d ch",
             SPELL_MEL_N_FFT, SPELL_MEL_N_MELS, SPELL_MFCC_N_COEFFS,
             SPELL_FEAT_N_FRAMES, SPELL_FEAT_N_CHANNELS);
    return ESP_OK;
}

void feat_extract_compute(const int16_t *pcm, int n_samples, float *out)
{
    const float floor_val = 1e-10f;
    const int   stride_t  = SPELL_MFCC_N_COEFFS * SPELL_FEAT_N_CHANNELS;
    const int   stride_k  = SPELL_FEAT_N_CHANNELS;

    // ─── Pass 1: PCM → MFCC ──────────────────────────────────────────
    for (int frame = 0; frame < SPELL_FEAT_N_FRAMES; frame++) {
        int start = frame * SPELL_MEL_HOP_LENGTH;

        // Load window into the FFT buffer with on-the-fly pre-emphasis.
        // y[n] = x[n] - α·x[n-1], with x[-1] = 0.
        for (int i = 0; i < SPELL_MEL_N_FFT; i++) {
            if (i < SPELL_MEL_WIN_LENGTH && (start + i) < n_samples) {
                float curr = (float)pcm[start + i] / 32768.0f;
                float prev = (start + i > 0)
                                ? (float)pcm[start + i - 1] / 32768.0f
                                : 0.0f;
                float emph = curr - SPELL_MFCC_PRE_EMPHASIS * prev;
                s_fft_buf[2 * i]     = emph * s_hann_window[i];
            } else {
                s_fft_buf[2 * i]     = 0.0f;
            }
            s_fft_buf[2 * i + 1] = 0.0f;
        }

        dsps_fft2r_fc32(s_fft_buf, SPELL_MEL_N_FFT);
        dsps_bit_rev_fc32(s_fft_buf, SPELL_MEL_N_FFT);

        // Power spectrum.
        float power[N_FFT_BINS];
        for (int k = 0; k < N_FFT_BINS; k++) {
            float re = s_fft_buf[2 * k];
            float im = s_fft_buf[2 * k + 1];
            power[k] = re * re + im * im;
        }

        // Mel filterbank → log → log_mel[40].
        float log_mel[SPELL_MEL_N_MELS];
        for (int m = 0; m < SPELL_MEL_N_MELS; m++) {
            float energy = 0.0f;
            const float *filt = &s_mel_filterbank[m * N_FFT_BINS];
            const int lo = s_mel_bounds[m].lo;
            const int hi = s_mel_bounds[m].hi;
            for (int k = lo; k < hi; k++) {
                energy += filt[k] * power[k];
            }
            log_mel[m] = logf(fmaxf(energy, floor_val));
        }

        // DCT-II orthonormal → MFCC[20]. Write directly into channel 0.
        for (int j = 0; j < SPELL_MFCC_N_COEFFS; j++) {
            float acc = 0.0f;
            const float *row = &s_dct_matrix[j * SPELL_MEL_N_MELS];
            for (int m = 0; m < SPELL_MEL_N_MELS; m++) {
                acc += row[m] * log_mel[m];
            }
            out[frame * stride_t + j * stride_k + 0] = acc;
        }
    }

    // ─── Pass 2: Δ MFCC ──────────────────────────────────────────────
    //
    // Δc[t,k] = (c[t+1,k] − c[t−1,k] + 2·(c[t+2,k] − c[t−2,k])) / 10
    // Edge boundaries: replicate (clamp t±n into [0, N-1]).
    //
    const int N = SPELL_FEAT_N_FRAMES;
    for (int t = 0; t < N; t++) {
        int t_p1 = t + 1; if (t_p1 > N - 1) t_p1 = N - 1;
        int t_p2 = t + 2; if (t_p2 > N - 1) t_p2 = N - 1;
        int t_m1 = t - 1; if (t_m1 < 0)     t_m1 = 0;
        int t_m2 = t - 2; if (t_m2 < 0)     t_m2 = 0;

        for (int k = 0; k < SPELL_MFCC_N_COEFFS; k++) {
            float c_p1 = out[t_p1 * stride_t + k * stride_k + 0];
            float c_p2 = out[t_p2 * stride_t + k * stride_k + 0];
            float c_m1 = out[t_m1 * stride_t + k * stride_k + 0];
            float c_m2 = out[t_m2 * stride_t + k * stride_k + 0];
            float delta = (c_p1 - c_m1 + 2.0f * (c_p2 - c_m2)) / 10.0f;
            out[t * stride_t + k * stride_k + 1] = delta;
        }
    }

    // ─── Pass 3: ΔΔ MFCC ─────────────────────────────────────────────
    for (int t = 0; t < N; t++) {
        int t_p1 = t + 1; if (t_p1 > N - 1) t_p1 = N - 1;
        int t_p2 = t + 2; if (t_p2 > N - 1) t_p2 = N - 1;
        int t_m1 = t - 1; if (t_m1 < 0)     t_m1 = 0;
        int t_m2 = t - 2; if (t_m2 < 0)     t_m2 = 0;

        for (int k = 0; k < SPELL_MFCC_N_COEFFS; k++) {
            float d_p1 = out[t_p1 * stride_t + k * stride_k + 1];
            float d_p2 = out[t_p2 * stride_t + k * stride_k + 1];
            float d_m1 = out[t_m1 * stride_t + k * stride_k + 1];
            float d_m2 = out[t_m2 * stride_t + k * stride_k + 1];
            float dd   = (d_p1 - d_m1 + 2.0f * (d_p2 - d_m2)) / 10.0f;
            out[t * stride_t + k * stride_k + 2] = dd;
        }
    }
}

void feat_extract_deinit(void)
{
    if (s_hann_window)    { heap_caps_free(s_hann_window);    s_hann_window    = NULL; }
    if (s_mel_filterbank) { heap_caps_free(s_mel_filterbank); s_mel_filterbank = NULL; }
    if (s_mel_bounds)     { heap_caps_free(s_mel_bounds);     s_mel_bounds     = NULL; }
    if (s_dct_matrix)     { heap_caps_free(s_dct_matrix);     s_dct_matrix     = NULL; }
    if (s_fft_buf)        { heap_caps_free(s_fft_buf);        s_fft_buf        = NULL; }
}
