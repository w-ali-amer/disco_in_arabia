"""exp42r CONFIRM driver -- doc 65 SS8.4, ordered 2026-09-03 ~03:00 (+03),
committed to the doc BEFORE the confirm dataset existed.

Fresh generator draw at seed 42000 (`stories_exp42r_confirm.json`, sha
asserted below; strong V-suite PASSED, `results_exp42r_confirm_validation
.json`). Arms A2 and B1 only, 5 seeds, registered init, plus
C1-on-confirm-A2 (phase 2 runs C1/C3/C4 unconditionally; C1 is the
required leg, C3/C4 are harmless extra controls). This driver patches
EXACTLY the dataset binding, its sha pin, and the output directories;
training, evaluation, controls and gate statistics are the committed
exp42r code paths, unforked.
"""
import os
import sys

CONFIRM_DATA = "stories_exp42r_confirm.json"
CONFIRM_SHA = ("472723fee82d81f9531a0263cc265e436fe7483b3f123d6d267de9dfe9"
               "ddcbed")
FIRST_DRAW_SHA = ("2326942ffb6ccf6bca95a7ec0502c82563957fabb159b143608a9fb"
                  "31f8f3113")

os.environ["EXP42R_CKPT_DIR"] = "exp42r_confirm_ckpt"
os.environ["EXP42R_CTL_DIR"] = "exp42r_confirm_ctl"
os.environ["EXP42R_LOCK_DIR"] = "exp42r_confirm_locks"

import json
meta = json.load(open(CONFIRM_DATA, encoding="utf-8"))["meta"]
got = meta.get("dataset_sha256") or meta.get("sha256")
assert got == CONFIRM_SHA, \
    "doc 65 SS8.4: confirm dataset sha mismatch (%r) -- STOP" % got
assert got != FIRST_DRAW_SHA, \
    "the confirm draw reproduced the FIRST draw -- it is not fresh"

import exp42r_common as K
K.DATA_PATH = CONFIRM_DATA
K.DATASET_SHA = CONFIRM_SHA
# `path=DATA_PATH` defaults were bound at def time; rebind them so every
# entry point resolves to the confirm draw (sha pin above is looked up
# dynamically inside load_dataset).
K.load_dataset.__defaults__ = (CONFIRM_DATA,)
K.provenance_gate.__defaults__ = (CONFIRM_DATA, True)

import exp42r_run as R

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--arms", "A2,B1"] + sys.argv[1:]
    sys.exit(R.main())
