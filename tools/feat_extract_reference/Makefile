# Pure-numpy reference of the firmware's MFCC + Δ + ΔΔ + normalization
# pipeline. Mirrors components/letter_recognizer/feat_extract.c step-for-
# step (modulo the FFT backend — numpy.fft.rfft vs esp-dsp Radix-2; both
# implement the same DFT and differ only at numerical-precision noise).
#
#   make selftest  # synthetic 1 kHz sine; checks shape + sanity + frame-79
#   make test      # alias for selftest
#
# To use against a real WAV, invoke the script directly:
#
#   python3 feat_extract_reference.py --wav letter_a.wav --out features.npy
#
# Optionally invoke the model (requires `pip install tflite-runtime` or
# `pip install tensorflow` — not installed by default to keep the host
# tools dependency-light):
#
#   python3 feat_extract_reference.py --wav letter_a.wav --invoke

PYTHON ?= python3
PROJECT_ROOT := $(abspath ../..)

.PHONY: all selftest test clean

all: selftest

selftest:
	cd $(PROJECT_ROOT) && $(PYTHON) tools/feat_extract_reference/feat_extract_reference.py --selftest

test: selftest

clean:
	rm -rf __pycache__
	@echo "(no host-build artifacts under tools/feat_extract_reference/)"
