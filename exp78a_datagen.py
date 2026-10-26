# -*- coding: utf-8 -*-
"""exp78a_datagen.py -- fresh confirm draw for exp78a at CONFIRM_SEED 77000.

Registration: qnlp_private_docs/68_exp78a_band32_replication.md §2.

Patches exp76_datagen.STORY_FMT and VALIDATION_OUT before calling its
main() at --seed 77000. Generates all 11 cells into confirm-named paths
and writes the V-suite report. No committed artifact is touched:
stories_exp76_*.json and results_exp76_WN_validation.json are never
read or written by this script.

Usage:
  python exp78a_datagen.py

Outputs:
  stories_exp78a_confirm_{cell}.json  (11 cells, at seed 77000)
  results_exp78a_WN_validation.json   (V-suite; must record kill_gate_passed: true)

FLAGGED AMBIGUITY (resolved toward committed code, per doc 63 standing rule):
I24_SEED = SEED + 24 in exp76_datagen.py binds at import time and stays
76024 even when main() sets SEED = 77000 globally. The I24 cells
(I24_n2_T2, I24_n4_T4) are therefore generated with the committed I24
inventory seed. Not patched: exp78a trains only band32_n2_T2 and its
S-shuffle floor (both use the I519 inventory, not affected); the V-suite
self-verifies all 11 cells by sha consistency regardless of which seed drew
the inventory.
"""
import sys

import exp76_datagen as D76

CONFIRM_SEED = 77000
CONFIRM_FMT = "stories_exp78a_confirm_%s.json"
CONFIRM_VSUITE = "results_exp78a_WN_validation.json"

# Patch BEFORE calling main().  exp76_datagen.main() does
# `global SEED; SEED = a.seed`, which controls I519 non-bearer pool
# (random.Random(SEED+7)), meta.seed and meta.is_confirm_draw.
# STORY_FMT and VALIDATION_OUT are NOT CLI-settable and must be patched here.
D76.STORY_FMT = CONFIRM_FMT
D76.VALIDATION_OUT = CONFIRM_VSUITE

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--seed", str(CONFIRM_SEED)]
    sys.exit(D76.main())
