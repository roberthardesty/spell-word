/**
 * @file feat_extract.h
 * @brief MFCC + Δ + ΔΔ feature extractor — produces the 3-channel input
 *        tensor the DS-CNN expects.
 *
 * Pipeline:
 *
 *   audio[12800] → pre-emphasis (0.97) → 80 × Hann-windowed frames →
 *   512-pt FFT → power → mel filterbank[40] → ln → DCT-II[20] →
 *   stack with Δ[20] and ΔΔ[20] across time → tensor[80, 20, 3]
 *
 * Output layout (NHWC, matches TFLite default):
 *
 *   out[t * 20 * 3 + k * 3 + c]
 *
 *   c = 0: MFCC coefficient k at frame t
 *   c = 1: first time-derivative (Δ) of MFCC k at frame t
 *   c = 2: second time-derivative (ΔΔ) of MFCC k at frame t
 *
 * Constants are read from spell_config.h. All choices that need to match
 * the training pipeline are constants there.
 *
 * Replaces the older log-mel-only front-end. The mel filterbank stage
 * still exists internally (MFCC = DCT(log(mel-energies))), but the public
 * API now produces MFCC + Δ + ΔΔ rather than raw log-mel.
 */

#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate workspace, build the mel filterbank and DCT matrix, initialize
 * esp-dsp twiddles. Must be called exactly once before feat_extract_compute().
 */
esp_err_t feat_extract_init(void);

/**
 * Compute the 3-channel feature tensor for a single utterance window.
 *
 * @param pcm        Input samples (int16, mono).
 * @param n_samples  Number of samples; should be SPELL_INFERENCE_WINDOW_SAMPLES.
 *                   Trailing samples are zero-padded by the framer.
 * @param out        SPELL_FEAT_N_ELEMENTS floats — caller-allocated. Layout
 *                   is [SPELL_FEAT_N_FRAMES][SPELL_MFCC_N_COEFFS]
 *                   [SPELL_FEAT_N_CHANNELS] flat row-major.
 */
void feat_extract_compute(const int16_t *pcm, int n_samples, float *out);

/**
 * Free workspace allocations. Safe to call without a prior init().
 */
void feat_extract_deinit(void);

#ifdef __cplusplus
}  // extern "C"
#endif
