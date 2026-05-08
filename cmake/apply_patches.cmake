# ─────────────────────────────────────────────────────────────────────────────
# Local-patch auto-applier for vendored / managed_components/ dependencies.
#
# Why this file exists
# --------------------
# managed_components/ is fetched by the IDF Component Manager and is gitignored
# in this repo. Any local fix to those vendored files (e.g. a forwarded bug fix
# we haven't gotten upstream yet) lives as a .patch under docs/patches/. To
# survive `idf.py reconfigure` / `idf.py fullclean` / a fresh checkout, those
# patches must be auto-applied as part of the build configure step. This file
# is the thing that does that.
#
# When this runs
# --------------
# Included from the top-level CMakeLists.txt AFTER `project(spell_word)`. By
# that point the IDF Component Manager has populated managed_components/ — so
# the target files exist and we can patch them before any compile step sees
# them.
#
# The contract for a patch
# ------------------------
# Every patch this system handles must include a unique MARKER string in the
# patched file (typically "Vikunja #<id>" embedded in a comment). The marker
# serves three purposes:
#   1. idempotency  — re-applying configure must not re-apply the patch
#   2. detection    — the build can tell "patched" from "pristine" without
#                     parsing diff context
#   3. auditability — `grep` can answer "is the fix in place?" in one second
#
# What happens on each invocation
# -------------------------------
#   marker present in target file  →  log "already applied", skip
#   target file does not exist     →  log a warning, skip (deps not fetched
#                                     yet, or upstream moved the file)
#   marker absent, patch applies   →  log "applied", continue
#   marker absent, patch fails     →  FATAL_ERROR with diagnostic context
#
# Failure mode → silent regression
# --------------------------------
# This system fails LOUDLY by design. A failed apply aborts the configure step
# rather than warning. The whole point of automation here is to prevent the
# 532 ms-invoke regression that motivated #53; a soft-warn would defeat that.
# If you legitimately need to bypass (e.g. during patch development), comment
# out the call site in the top-level CMakeLists.txt — do NOT relax the apply
# logic here.
#
# Adding a new patch
# ------------------
# 1. Generate a unified diff (`diff -u` or `git diff --no-prefix` then prepend
#    a/  b/  to paths so `patch -p1` works from the repo root).
# 2. Embed a unique marker in the patched file (typically a comment containing
#    "Vikunja #<id>" — match the style in the existing patch header).
# 3. Save under docs/patches/<id>-<short-name>.patch with a header like the
#    existing 53-* patch — explain why, what, how-to-apply-by-hand.
# 4. Add an apply_repo_patch() call in the top-level CMakeLists.txt below the
#    existing one.
# 5. Run `idf.py reconfigure` to verify the auto-apply fires.
# ─────────────────────────────────────────────────────────────────────────────

function(apply_repo_patch)
    set(_options)
    set(_one_value LABEL TARGET_FILE MARKER PATCH_FILE)
    set(_multi_value)
    cmake_parse_arguments(P "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})

    if(NOT P_LABEL OR NOT P_TARGET_FILE OR NOT P_MARKER OR NOT P_PATCH_FILE)
        message(FATAL_ERROR
            "[patches] apply_repo_patch() requires LABEL, TARGET_FILE, "
            "MARKER, and PATCH_FILE.")
    endif()

    if(NOT EXISTS "${P_PATCH_FILE}")
        message(FATAL_ERROR
            "[patches] ${P_LABEL}: patch file is missing: ${P_PATCH_FILE}")
    endif()

    if(NOT EXISTS "${P_TARGET_FILE}")
        # The target may not exist on a brand-new clone before deps fetch;
        # but project() should have fetched by the time we get here. If it's
        # still missing the upstream dependency may have moved the file —
        # warn but don't fail, since some dev workflows (host-only `make
        # test` etc.) don't need the vendored fix.
        message(WARNING
            "[patches] ${P_LABEL}: target file missing — skipping apply.\n"
            "  expected: ${P_TARGET_FILE}\n"
            "  Possible causes:\n"
            "    1. managed_components/ has not been populated yet (run\n"
            "       `idf.py reconfigure` to trigger the component manager)\n"
            "    2. the upstream dependency was bumped and renamed/moved\n"
            "       this file — refresh the patch against the new layout.")
        return()
    endif()

    file(READ "${P_TARGET_FILE}" _target_contents)
    string(FIND "${_target_contents}" "${P_MARKER}" _marker_pos)
    if(_marker_pos GREATER_EQUAL 0)
        message(STATUS
            "[patches] ${P_LABEL}: already applied (marker found in target).")
        return()
    endif()

    message(STATUS "[patches] ${P_LABEL}: applying ${P_PATCH_FILE}")
    execute_process(
        COMMAND patch -p1 --forward -i "${P_PATCH_FILE}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _patch_status
        OUTPUT_VARIABLE _patch_stdout
        ERROR_VARIABLE  _patch_stderr
    )

    if(NOT _patch_status EQUAL 0)
        message(FATAL_ERROR
            "\n[patches] ${P_LABEL}: APPLY FAILED (exit ${_patch_status}).\n"
            "  patch file:  ${P_PATCH_FILE}\n"
            "  target:      ${P_TARGET_FILE}\n"
            "  stdout:\n${_patch_stdout}\n"
            "  stderr:\n${_patch_stderr}\n"
            "Most common cause: the upstream dependency was bumped and the\n"
            "diff context lines no longer match. Open the .patch file and\n"
            "the target file side by side, adjust the context, then re-run\n"
            "`idf.py reconfigure`. See docs/patches/README.md for the full\n"
            "diagnosis playbook.")
    endif()

    # Defense in depth — if patch reported success but the marker is somehow
    # still absent, the patch is structurally broken (e.g. header diff, no-op
    # hunk). Surface that immediately rather than letting it ride.
    file(READ "${P_TARGET_FILE}" _target_post)
    string(FIND "${_target_post}" "${P_MARKER}" _post_pos)
    if(_post_pos LESS 0)
        message(FATAL_ERROR
            "[patches] ${P_LABEL}: patch reported success but marker "
            "'${P_MARKER}' is still absent in ${P_TARGET_FILE}. The patch is "
            "structurally invalid — inspect by hand.")
    endif()

    message(STATUS "[patches] ${P_LABEL}: applied successfully.")
endfunction()
