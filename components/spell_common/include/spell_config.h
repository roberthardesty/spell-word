/**
 * @file spell_config.h
 * @brief Single source of truth for all firmware tunables and pin assignments.
 *
 * IMPORTANT: many of these constants must match the model-training pipeline
 * exactly. Sample rate, FFT size, hop, window, n_mels, fmin/fmax, the energy
 * gate threshold, the inference window length — any drift between firmware
 * and training silently degrades on-device accuracy. The training tool should
 * ingest this header (or a generated JSON sidecar) and reference the same
 * values.
 *
 * See docs/MVP-Implementation-Plan.md §5 for the full discussion.
 */

#pragma once

// =============================================================================
// Pin assignments (DevKitC-1 bring-up; revise for custom PCB)
// =============================================================================

// I2S RX — ICS-43434 MEMS microphone (24-bit two's-complement in 32-bit slot).
// SEL pin must be tied to GND on the breakout so the mic drives LEFT slot.
#define SPELL_I2S_RX_PORT             I2S_NUM_0
#define SPELL_I2S_RX_SD_PIN           4   // DOUT from mic
#define SPELL_I2S_RX_SCK_PIN          5   // BCLK
#define SPELL_I2S_RX_WS_PIN           6   // LRCLK / WS

// I2S TX — MAX98357A I2S DAC + class-D amp driving the speaker.
// The amp's GAIN pin sets fixed gain; the SD pin (FW_AMP_SHUTDOWN_GPIO)
// gates output to suppress idle hiss between playback events.
#define SPELL_I2S_TX_PORT             I2S_NUM_1
#define SPELL_I2S_TX_SD_PIN           7   // DIN to amp
#define SPELL_I2S_TX_SCK_PIN          15  // BCLK
#define SPELL_I2S_TX_WS_PIN           16  // LRC

// Controls.
#define SPELL_BUTTON_GPIO             9   // active-low momentary, internal pull-up
#define SPELL_AMP_SHUTDOWN_GPIO       10  // pull low to mute amp when idle

// Status LED — DevKitC-1 onboard WS2812 is on GPIO 48; we treat it as a
// plain digital output for MVP, replace with a proper RGB driver later.
#define SPELL_STATUS_LED_PIN          48

// Factory reset pin — BOOT button (GPIO 0). Hold at power-on for 5 s to
// erase NVS and any user-tunable state. Inherited from the EARS pattern.
#define SPELL_FACTORY_RESET_PIN       0
#define SPELL_FACTORY_RESET_HOLD_MS   5000

// =============================================================================
// Audio format
// =============================================================================

#define SPELL_SAMPLE_RATE             16000   // Hz
#define SPELL_BITS_PER_SAMPLE         16
#define SPELL_CHANNELS                1
#define SPELL_BYTES_PER_SAMPLE        (SPELL_BITS_PER_SAMPLE / 8)

// =============================================================================
// Audio capture geometry
// =============================================================================

// 256 frames @ 16 kHz = 16 ms per i2s_channel_read(). Trade between syscall
// overhead and tap latency — same value as the EARS POC.
#define SPELL_CAPTURE_CHUNK_FRAMES    256

// Max concurrent audio_capture subscribers. For MVP we have 1 (segmenter);
// debug audio dump and future taps share this pool.
#define SPELL_CAPTURE_MAX_SUBS        4

// =============================================================================
// Inference window (DS-CNN expects 800 ms @ 16 kHz)
// =============================================================================
//
// 800 ms window matches the training pipeline's max_duration_ms (see the
// training-project config.yaml). Letters longer than this are tail-clipped
// in the segmenter; W and similar multi-syllable letters fit comfortably.

#define SPELL_INFERENCE_WINDOW_SAMPLES   12800                                   // 800 ms
#define SPELL_INFERENCE_WINDOW_BYTES \
    (SPELL_INFERENCE_WINDOW_SAMPLES * SPELL_BYTES_PER_SAMPLE)                    // 25 600

// Segmenter's StreamBuffer subscriber size: one full inference window plus
// one capture chunk of slack to absorb scheduling jitter.
#define SPELL_SEGMENTER_SUB_BYTES \
    (SPELL_INFERENCE_WINDOW_BYTES + \
     SPELL_CAPTURE_CHUNK_FRAMES * (int)sizeof(int16_t))                          // 26 112

// =============================================================================
// Feature front-end (must match training pipeline exactly)
// =============================================================================
//
// Pipeline:  pre-emphasis → frame+window → FFT → mel filterbank → log →
//            DCT-II (orthonormal) → MFCC[20] →
//            stack with Δ and ΔΔ → 3-channel feature tensor [79, 20, 3].
//
// Constants below are sourced from the training project's config.yaml. Any
// drift here silently degrades on-device accuracy; the training tool should
// either ingest this header or be cross-checked against it on every export.

// --- Mel filterbank stage ---
#define SPELL_MEL_N_FFT               512
#define SPELL_MEL_HOP_LENGTH          160     // 10 ms hop  @ 16 kHz
#define SPELL_MEL_WIN_LENGTH          320     // 20 ms window @ 16 kHz
#define SPELL_MEL_N_MELS              40      // intermediate filterbank bands
#define SPELL_MEL_FMIN                50.0f   // Hz
#define SPELL_MEL_FMAX                7600.0f // Hz

// --- MFCC + Δ + ΔΔ stage ---
#define SPELL_MFCC_N_COEFFS           20      // top-N DCT coefficients
#define SPELL_MFCC_PRE_EMPHASIS       0.97f   // y[n] = x[n] - α·x[n-1]
#define SPELL_FEAT_N_CHANNELS         3       // MFCC + Δ + ΔΔ
#define SPELL_FEAT_N_FRAMES           80      // training pipeline produces 80
                                              // frames per 800 ms window
                                              // (audio padded to 12960 samples
                                              // before framing with hop=160,
                                              // win=320). The firmware framer
                                              // covers samples [t·160, t·160+320)
                                              // and zero-fills any out-of-bounds
                                              // tail, so producing 80 frames
                                              // from a 12800-sample buffer
                                              // matches the training byte-for-
                                              // byte at every in-bounds index
                                              // and at the trailing zero-pad.
                                              // Bumped from 79 in Vikunja #52
                                              // to match the model's input
                                              // shape [1, 80, 20, 3].

// Convenience: total floats in the feature tensor for one window.
#define SPELL_FEAT_N_ELEMENTS \
    (SPELL_FEAT_N_FRAMES * SPELL_MFCC_N_COEFFS * SPELL_FEAT_N_CHANNELS)         // 4 800

// =============================================================================
// Energy gate
// =============================================================================

// Inference-time energy gate. Should match the threshold used by the
// training-pipeline curation tool: feeding the model audio quieter than it
// saw during training produces out-of-distribution garbage.
#define SPELL_ENERGY_GATE_DB          (-45.0f)

// =============================================================================
// VAD / segmenter
// =============================================================================

#define SPELL_VAD_ON_DBFS             (-38.0f)   // letter onset (more permissive than gate)
#define SPELL_VAD_OFF_DBFS            (-42.0f)   // letter offset
#define SPELL_VAD_OFF_FRAMES          5          // ~80 ms below off threshold
#define SPELL_VAD_MIN_LETTER_MS       150        // reject sub-150 ms blips
#define SPELL_VAD_MIN_GAP_MS          80         // merge if next onset within 80 ms
#define SPELL_VAD_END_OF_WORD_MS      1200       // sustained silence → word complete
#define SPELL_VAD_MAX_LETTER_MS       800        // force split if a letter exceeds this

// Pre-roll ring buffer size (in samples). The segmenter keeps the most recent
// SPELL_VAD_PREROLL_SAMPLES of audio so a detected onset can include
// pre-onset context — the DS-CNN was trained on 600 ms windows that include
// the leading edge of the letter, not just the high-energy core.
#define SPELL_VAD_PREROLL_SAMPLES     3200       // 200 ms @ 16 kHz

// =============================================================================
// Inference / TFLite Micro
// =============================================================================

#define SPELL_LETTER_TOP_K            5
// Tensor arena AND model flatbuffer both live in internal SRAM
// (MALLOC_CAP_INTERNAL). PSRAM placement is 3-5× slower for TFLM
// activations (arena) and even worse for weights (model), since every
// Conv2D/DepthwiseConv2D MAC reads weights from the flatbuffer — see the
// comment block at the head of letter_recognizer.cpp.
//
// Sizing on ESP32-S3 with 8 MB octal PSRAM and the production partition
// table:
//   - heap_init reports ~323 KiB internal DRAM
//   - largest contiguous internal block at recognizer-init time (after
//     FreeRTOS task stacks, USB-Serial-JTAG driver, audio_capture, and
//     feat_extract pre-allocs) is ~252 KiB on 2026-05-07 hardware
//
// Measurements with the 53 KiB flat26_baseline_int8 model
// (input [1,80,20,3] INT8, output [1,26] INT8):
//   - arena_used_bytes = 100,920 B (98.5 KiB)
//   - invoke time, model in PSRAM: 532 ms (blows ADR-0001 500 ms floor)
//   - invoke time, model in internal SRAM: <see boot log on next flash>
//
// Historical bench notes (kept for review):
//   - 256 KiB arena alone fails the internal alloc
//   - 192 KiB arena alone allocates but starves USB-Serial-JTAG so the
//     device soft-bricks (BOOT-button ROM recovery required)
//   - 144 KiB arena + PSRAM model: works, but 532 ms invoke
//   - 118 KiB arena + 64 KiB internal model: target — 30-40 ms invoke
//
// SPELL_TENSOR_ARENA_SIZE is sized at ≤ 1.2 × measured arena_used
// (100,920 × 1.2 = 121,104 → 118 KiB = 120,832 B is the largest 1 KiB
// multiple under the cap, leaving 17.7 KiB headroom for any kernel
// scratch growth from esp-tflite-micro version bumps).
//
// SPELL_MODEL_INTERNAL_BUF_SIZE caps the internal-SRAM model copy. The
// model partition is 256 KiB on flash, but the live flatbuffer is ~53
// KiB; we only mirror the first 64 KiB into internal SRAM. If a future
// model bumps over 64 KiB, AllocateTensors will fail (the flatbuffer's
// internal pointers walk past the truncated buffer) — bump the cap and
// re-bench. If the internal alloc fails outright (memory pressure),
// load_model_from_partition() falls back to PSRAM and logs a warning.
#define SPELL_TENSOR_ARENA_SIZE       (118 * 1024)
#define SPELL_MODEL_INTERNAL_BUF_SIZE (64  * 1024)
#define SPELL_MODEL_PARTITION_LABEL   "model"
#define SPELL_INFERENCE_TASK_STACK    8192       // bytes; heavy buffers in PSRAM heap

// Decoder partition labels (subtypes 0x81 / 0x82 in partitions.csv). The
// label strings are descriptive in-CSV; only the subtypes are load-bearing.
#define SPELL_MATRIX_PARTITION_LABEL  "confusion"
#define SPELL_DICT_PARTITION_LABEL    "dictionary"

// Inference utterance queue depth. Letters are emitted at human pace
// (~3-5 per word, with gaps), so 16 is generous. Overflow drops oldest.
#define SPELL_INFERENCE_QUEUE_DEPTH   16

// =============================================================================
// Decoder (confusion-aware word resolution)
// =============================================================================

#define SPELL_DECODER_ALPHA           0.15f      // network softmax vs. confusion matrix mix
#define SPELL_DECODER_LAMBDA          0.5f       // Zipfian frequency prior weight
#define SPELL_EDIT_INS_PENALTY        (-2.0f)    // spurious utterance (cough)
#define SPELL_EDIT_DEL_PENALTY        (-2.5f)    // missed letter (swallowed pause)
#define SPELL_DECISION_MARGIN         2.0f       // log-prob margin for non-abstention
#define SPELL_MAX_LETTERS_PER_WORD    16         // bound on per-word top-K buffer

// Aggressive early-commit predicate (see PRD 0001 §"Aggressive early-commit
// predicate" and ADR-0001). The decoder commits early on inter-letter silence
// once all three clauses pass. The 500 ms silence floor is load-bearing — it
// also serves as the natural barrier against the W-detection race (one
// re-inference cycle is ~40 ms).
#define SPELL_EARLY_MIN_SILENCE_MS    500        // post-letter silence floor
#define SPELL_EARLY_MARGIN_RUNNERUP   3.0f       // log-prob over same-length runner-up
#define SPELL_EARLY_MARGIN_LONGER     2.0f       // log-prob over best longer-by-1-or-2

// =============================================================================
// W detector
// =============================================================================
//
// Trigger thresholds for "recognize W as a single letter" via inference-result
// pattern matching (ADR-0005, PRD 0001 §"W-detection trigger"). Defaults are
// best-guess; Phase 7 sweeps tune them against real recordings.

#define SPELL_W_DETECT_U_THRESHOLD          0.50f  // current top-1 'U' must clear this
#define SPELL_W_DETECT_LOW_CONF_THRESHOLD   0.40f  // each of the prior two below this
#define SPELL_W_DETECT_CONFIRM_THRESHOLD    0.70f  // re-inference top-1 'W' must clear this

// PCM ring depth in letter_recognizer: 3 candidate utterances in the detection
// window plus 1 currently being processed.
#define SPELL_W_DETECT_PCM_RING_DEPTH       4

// =============================================================================
// UI / state machine
// =============================================================================

#define SPELL_BUTTON_DEBOUNCE_MS      30
#define SPELL_ARM_TIMEOUT_MS          10000      // auto-disarm if no audio

// Tones synthesised at boot (`spell_ui` materialises these). IDs are stable so
// the UI core can encode actions as data; frequencies are tunable.
#define SPELL_TONE_CONFIRMATION       0          // press → ARMED
#define SPELL_TONE_RESOLVED           1          // word resolved → "got it"
#define SPELL_TONE_ERROR_CHIRP        2          // abstain → try again

#define SPELL_TONE_CONFIRMATION_HZ    880        // A5
#define SPELL_TONE_RESOLVED_HZ        1320       // ~E6
#define SPELL_TONE_ERROR_CHIRP_HZ     220        // A3
