/**
 * @file audio_dump.h
 * @brief Continuous circular capture of recent PCM, dumpable to UART.
 *
 * Used to bring back recordings of failed spellings for offline VAD replay.
 * The use case: child spells a word, the device gets it wrong, you press a
 * debug button (or hit a serial command), and the device emits the last
 * ~10 s of audio over UART. A host-side script (tools/serial_to_wav.py)
 * captures the framed output and converts it back to a 16 kHz mono WAV
 * which you can replay through tools/segmenter_replay.
 *
 * Lifecycle:
 *
 *   1. audio_dump_init(buffer_seconds)
 *      MUST be called between audio_capture_init() and audio_capture_start().
 *      Allocates a PSRAM ring buffer sized for buffer_seconds × 32 KB and
 *      spawns a drain task on Core 0.
 *   2. The drain task continuously copies PCM from the StreamBuffer
 *      subscription into the ring, overwriting oldest samples.
 *   3. audio_dump_emit_b64(seconds) is called from a debug context
 *      (a button handler, a console command, an event handler). It freezes
 *      the drain, base64-encodes the most recent @p seconds of audio, and
 *      writes a framed payload to stdout (which is UART).
 *
 * Output format on UART:
 *
 *      ===AUDIO_DUMP_BEGIN n_samples=NNN rate=16000===
 *      <base64 line, 76 chars wide>
 *      ...
 *      ===AUDIO_DUMP_END===
 *
 * The base64 payload is raw little-endian int16 PCM. The receiver script
 * adds the WAV header.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate the PSRAM ring buffer (buffer_seconds × 32 KB) and spawn the
 * drain task. The ring overwrites oldest samples once full.
 *
 * Must be called between audio_capture_init() and audio_capture_start().
 *
 * Reasonable @p buffer_seconds values: 5–15 s. Default in app_main = 10 s.
 */
esp_err_t audio_dump_init(int buffer_seconds);

/**
 * Emit the most recent @p seconds of buffered audio to UART as
 * base64-framed PCM. Pass -1 to emit the entire buffer.
 *
 * Briefly freezes the drain task so the snapshot is consistent. The drain
 * may drop a small number of samples while frozen — this is logged but is
 * not fatal (capture itself is not interrupted; the StreamBuffer's
 * overrun_count just ticks up).
 *
 * Safe to call from any task context. Not safe to call from an ISR.
 */
esp_err_t audio_dump_emit_b64(int seconds);

/**
 * @return number of seconds currently buffered (≤ buffer_seconds).
 *         Useful for "wait until at least N seconds are available before
 *         dumping" patterns.
 */
int audio_dump_buffered_seconds(void);

#ifdef __cplusplus
}  // extern "C"
#endif
