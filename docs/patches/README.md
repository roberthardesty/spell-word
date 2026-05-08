# Local patches against managed_components/

This directory holds unified-diff patches that the build system applies to
files inside `managed_components/` at CMake configure time. It exists because
`managed_components/` is fetched by the IDF Component Manager and is
gitignored — any local fix to a vendored file would otherwise be silently
overwritten by the next `idf.py reconfigure`.

## How it works

```
                    ┌─────────────────────────────┐
  idf.py build  ──▶ │ cmake -B build              │
                    │   include(project.cmake)    │
                    │   project(spell_word)       │ ← Component Manager
                    │                             │   fetches deps into
                    │                             │   managed_components/
                    │   include(apply_patches)    │
                    │   apply_repo_patch(...)     │ ← THIS DIR's patches
                    │                             │   applied here
                    │   register components       │
                    │   …                         │
                    └─────────────────────────────┘
                                 │
                                 ▼
                    compile sees patched files
```

The `apply_repo_patch()` function lives in `cmake/apply_patches.cmake`. It is
called from the top-level `CMakeLists.txt` once per patch, **after**
`project()` (so deps are fetched) and **before** any compile step (so the
patch lands in time).

## The contract for a patch

Every patch in this directory must satisfy three properties:

1. **Be a unified diff applicable from the repo root with `patch -p1`.**
   Use `--no-prefix` style with `a/`+`b/` if generating with `git diff`. The
   existing `53-esp-nn-conv-dispatcher-guard.patch` is the reference shape.
2. **Embed a unique MARKER string in the patched file.** Convention:
   `Vikunja #<id>` inside a comment near the changed lines. The marker is
   how the build detects "already applied" without trying to dry-run the
   patch — fast, exact, greppable.
3. **Carry a header explaining why, what, and how to re-apply by hand.**
   The header is read by humans coming off a context switch, so optimize
   for that audience: name the upstream version, link the Vikunja issue,
   list the failure mode if the patch isn't applied.

## What the build does on each `idf.py reconfigure`

For each registered patch, `apply_repo_patch()` does:

| state | action | log |
|---|---|---|
| marker present in target file | skip | `[patches] <label>: already applied (marker found in target).` |
| target file does not exist | skip with warning | warns; build continues (host-only flows don't need vendored fixes) |
| marker absent, patch applies cleanly | apply, then re-verify marker | `[patches] <label>: applying …` then `applied successfully.` |
| marker absent, patch fails to apply | **`FATAL_ERROR`** | full stdout + stderr + path context |
| marker absent, patch reports success but marker still missing afterwards | **`FATAL_ERROR`** | structurally invalid patch |

The system is designed to fail **loudly**. A soft warning would defeat the
whole point — the patches in here exist because someone proved that without
them, real performance regressions slip through silently (cf. #53: 532 ms
invoke vs 84 ms with the fix).

## If you see `[patches] … APPLY FAILED` at configure time

The most common cause is an **upstream dependency bump**: the new file in
`managed_components/` has different context lines than what the patch was
generated against, so `patch` can't find the hunks.

Diagnosis playbook:

1. **Read the FATAL_ERROR output carefully.** It includes patch stdout
   ("Hunk #1 FAILED at line N", "patch unexpectedly ends in middle of
   line", etc.) which usually pinpoints the issue.
2. **Diff the upstream file against the version the patch was generated
   against.** The patch header (top of the .patch file) names the upstream
   version. Check `managed_components/<dep>/idf_component.yml` for the
   currently-fetched version.
3. **Decide:**
   - If the upstream fix is *still needed* — regenerate the patch against
     the new upstream, keeping the same marker string. Replace the file in
     `docs/patches/`.
   - If the upstream fixed the underlying defect — delete the .patch file
     and the corresponding `apply_repo_patch()` call in `CMakeLists.txt`.
4. **Re-run `idf.py reconfigure`** and confirm the configure step logs
   `applied successfully.` (or the patch is no longer registered).

## If you need to bypass auto-apply temporarily

Use cases: developing a new patch, comparing pristine-vs-patched performance,
debugging a kernel that interacts with the patched code.

Comment out the `apply_repo_patch()` call in the top-level `CMakeLists.txt`,
manually revert the target file (`patch -R -p1 -i docs/patches/<file>`), and
run `idf.py reconfigure`. **Do not relax the apply logic in
`cmake/apply_patches.cmake`** — that file's job is to fail loudly, and weakening
it weakens every patch.

## Patch lifecycle — when to remove

A patch should leave this directory the moment its underlying defect is
fixed upstream and the dependency is bumped past that fix. Keeping a stale
patch around past that point is harmless until the upstream code changes
again, at which point it'll start failing to apply and someone will spend
an hour debugging a fix that's no longer needed.

For each patch, periodically check:

- Is the upstream issue (linked in the patch header) closed?
- Has the dependency in `idf_component.yml` been bumped past that fix?
- If yes to both: bump our pin (or remove the version constraint), remove
  the .patch file, remove the `apply_repo_patch()` call.

## Currently registered patches

| file | target | upstream issue | landed |
|---|---|---|---|
| `53-esp-nn-conv-dispatcher-guard.patch` | `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` | not yet filed upstream | Vikunja #53, 2026-05-07 |
