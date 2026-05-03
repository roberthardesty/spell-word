# Production firmware is fully offline

The shipped device firmware exposes no network interfaces — no WiFi, no BLE, no upload paths, no telemetry. All test/debug instrumentation (PSRAM-resident audio capture, WiFi upload of utterance + inference + decoder + segmenter state for offline analysis) ships as a separate Kconfig-gated build target, not as runtime-configurable behavior. This forces a hard separation between "device behavior" and "test-rig behavior" rather than letting debug instrumentation leak into shipped firmware. Consequence: model and corpus partition updates use USB (`parttool.py`) for MVP; an OTA mechanism is a v1 design question.

## Considered options

- **Runtime-toggled telemetry.** Rejected: a flag that *can* be flipped at runtime is a flag that *will* be flipped on shipped devices. Compile-time gating is the only way to make "no network" a guarantee.
- **Always-on telemetry behind a "developer mode" UX.** Rejected for a children's reading aid: the privacy story is a product asset, not a developer-experience axis.
