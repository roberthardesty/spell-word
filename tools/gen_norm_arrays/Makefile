# Host-side regenerator for components/letter_recognizer/feat_norm.{c,h}.
#
# Reads the .npy norm arrays from models/ and emits the C arrays that the
# firmware compiles in. The generated .c/.h are checked in so the IDF build
# does not depend on Python — re-run this when the model handoff changes.
#
#   make gen       # regenerate components/letter_recognizer/feat_norm.{c,h}
#   make check     # generate + round-trip self-check (atol=1e-7 vs .npy)
#   make test      # alias for `make check`

PYTHON ?= python3
PROJECT_ROOT := $(abspath ../..)

.PHONY: all gen check test clean

all: gen

gen:
	cd $(PROJECT_ROOT) && $(PYTHON) tools/gen_norm_arrays/gen_norm_arrays.py

check:
	cd $(PROJECT_ROOT) && $(PYTHON) tools/gen_norm_arrays/gen_norm_arrays.py --check

test: check

clean:
	@echo "(no host-build artifacts under tools/gen_norm_arrays/ — generated files live under components/letter_recognizer/)"
