/**
 * @file log_mel.h — DEPRECATED
 *
 * Superseded by feat_extract.h. The Spell-Word DS-CNN consumes a 3-channel
 * MFCC + Δ + ΔΔ tensor, not raw log-mel; the front-end was rebuilt
 * accordingly. See components/letter_classifier/feat_extract.{c,h} for
 * the live implementation.
 *
 * This header is kept in the tree only to preserve git history. It is no
 * longer included by any production code and the corresponding log_mel.c
 * is no longer compiled (see CMakeLists.txt). Delete both files when
 * you're confident nothing in your branches still depends on them.
 */

#pragma once
#error "log_mel.h is deprecated — include feat_extract.h instead."
