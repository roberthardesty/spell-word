/**
 * @file wav.h
 * @brief Minimal WAV reader/writer — 16-bit PCM, mono, 16 kHz only.
 *
 * Anything else is rejected with a clear error message. We don't need a
 * full WAV decoder; the firmware only ever consumes 16-bit mono at 16 kHz.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Read a 16-bit mono 16 kHz WAV file into a malloc'd int16_t buffer.
 * On success: *out_samples points to the buffer, *out_n is the sample
 * count, *out_rate is set to the file's actual rate (verified to be 16000),
 * and the caller must free(*out_samples).
 *
 * Returns 0 on success, -1 on any error (wrong format, can't open, etc.).
 * Prints a description of the failure to stderr.
 */
int wav_read_16k_mono(const char *path,
                      int16_t **out_samples,
                      size_t   *out_n,
                      int      *out_rate);

/**
 * Write a 16-bit mono 16 kHz WAV file. Same constraints (assumes 16 kHz).
 * Returns 0 on success, -1 on error.
 */
int wav_write_16k_mono(const char *path,
                       const int16_t *samples,
                       size_t n);

#ifdef __cplusplus
}  // extern "C"
#endif
