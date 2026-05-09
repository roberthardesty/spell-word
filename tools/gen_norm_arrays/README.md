# gen_norm_arrays

Generates `components/letter_recognizer/feat_norm.{c,h}` from
`models/norm_mean.npy` + `models/norm_std.npy`.

The TFLM int8 model in `models/flat26_baseline_int8.tflite` was calibrated
against MFCC features normalized per `(mfcc[k][c] - mean[k][c]) / std[k][c]`,
where `mean` and `std` are NumPy arrays of shape `(1, 1, 20, 3)` (60 floats
each). The firmware applies the normalization between
`feat_extract_compute()` and the int8 quantize loop in
`letter_recognizer.cpp`, reading the arrays from this generator's output.

The generated `.c` is checked in so the IDF build doesn't need Python at
compile time. Re-run the generator whenever `models/norm_*.npy` changes.

## Usage

From the project root:

```sh
# Regenerate components/letter_recognizer/feat_norm.{c,h}.
make -C tools/gen_norm_arrays gen

# Same, plus a self-check that re-parses the emitted .c and confirms the
# values match the .npy inputs to within 1e-7 (catches formatter drift).
make -C tools/gen_norm_arrays check
```

The script is idempotent — running it twice in a row produces a byte-identical
diff.

## Why a host script vs. building from .npy in CMake

The IDF build is cross-compiled and shouldn't pull in Python or NumPy at
configure time. The script runs once on the developer host and the result
is checked in. The same pattern is used for the per-component `_replay`
host CLIs — they're host tools that produce artifacts the firmware later
consumes.
