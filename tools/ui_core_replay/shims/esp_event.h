/**
 * @file shims/esp_event.h
 * @brief Host-build stub for the bits of esp_event.h that spell_events.h needs.
 *
 * spell_events.h declares event bases via ESP_EVENT_DECLARE_BASE — that is
 * the only esp_event symbol it touches. ui_core.h re-includes
 * spell_events.h purely to pick up `spell_ui_state_t`, so we just need the
 * macro to expand cleanly under the host compiler. None of the bases are
 * used by ui_core itself.
 *
 * This shim is found via `-Ishims` only when building tools/ui_core_replay/
 * tests; on-device the real esp_event.h shadows it.
 */

#pragma once

typedef const char *esp_event_base_t;

#define ESP_EVENT_DECLARE_BASE(name) extern esp_event_base_t name
