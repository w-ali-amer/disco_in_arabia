# -*- coding: utf-8 -*-
"""exp78a_comparator.py -- comparator for exp78a, band32_n2_T2 only.

Registration: qnlp_private_docs/68_exp78a_band32_replication.md §2, §3(b).

Thin driver over the committed exp76_comparator.py, patching ONLY:
  * dataset binding (D76.STORY_FMT -> confirm-named paths)
  * arm scope (R.ANCHOR_ARMS = (), R.TRAINED_CELLS = ["band32_n2_T2"])
  * output dir  (--out exp78a_ckpt/comparator.json)

Three tiers (i_raw, ii_committed, iii_distractor) on the fresh draw for
the band32_n2_T2 cell and its S-shuffle floor.

FLAGGED AMBIGUITY (resolved toward committed code): running all 24 arms
would exceed exp78a's registered scope.  R.ANCHOR_ARMS is set to () and
R.TRAINED_CELLS to ["band32_n2_T2"].  The flat-in-n check becomes
vacuous-null (one cell, one group) -- this is the coded behavior, not an
error.  Flagged per doc 63 standing rule.

Patch order (load-bearing):
  1. D76.STORY_FMT patched before any import of exp76_run (load_store uses it).
  2. R.ANCHOR_ARMS and R.TRAINED_CELLS patched before exp76_comparator import
     so the comparator's module-level `arms` list (list(R.ANCHOR_ARMS) +
     R.TRAINED_CELLS) is scoped to band32 only.
  3. OUT in exp76_comparator binds at import from R.CKPT_DIR (still
     "exp76_ckpt"); override via --out instead.

Usage:
  python exp78a_comparator.py
Writes:
  exp78a_ckpt/comparator.json
"""
import sys

CONFIRM_FMT    = "stories_exp78a_confirm_%s.json"
CONFIRM_VSUITE = "results_exp78a_WN_validation.json"
CONFIRM_CKPT   = "exp78a_ckpt"
CONFIRM_OUT    = CONFIRM_CKPT + "/comparator.json"

# ---- patch D76 before importing exp76_run (load_store reads STORY_FMT) --------
import exp76_datagen as D76
D76.STORY_FMT    = CONFIRM_FMT
D76.VALIDATION_OUT = CONFIRM_VSUITE

import exp76_run as R
R.EXP76_VSUITE  = CONFIRM_VSUITE   # belt-and-suspenders

# Restrict arms to band32 only (doc 68's registered scope).
# R.FLOOR_OF["band32_n2_T2"] already equals "S-shuffle-band32_n2_T2"
# and is unchanged.  No floor-arm rebinding needed.
R.ANCHOR_ARMS   = ()
R.TRAINED_CELLS = ["band32_n2_T2"]

# exp76_comparator.OUT binds at import from R.CKPT_DIR ("exp76_ckpt");
# we redirect via --out, so R.CKPT_DIR need not be patched.
import exp76_comparator as C

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--out", CONFIRM_OUT]
    sys.exit(C.main())
