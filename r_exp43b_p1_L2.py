# -*- coding: utf-8 -*-
"""r_exp43b_p1_L2.py -- doc 25 SSR.4 "Extended": exp43b Phase 1 on L2r only
(A3 + B1, the doc-23 grid, the any-epoch early-stop gate). Decides
`rung_learnable_L2_r` -- the 1-qubit reference for any "crossing" reading.

WRAPS exp43b_phase1.py (never edited). Rebinds:
  RUNGS      ("L1", "L2") -> ("L2r",)  (the runner iterates this module global)
  load_rung  "L2r" -> stories_exp43b_L2r.json under the pinned sha (the
             runner's own meta-sha check is preserved inside)
  CKPT_DIR   exp43b_ckpt -> exp43b_p1_r_ckpt (keys become L2r_*)
  fresh_results_path  results_exp43b_phase1 -> results_exp43b_p1_L2r (Rule 3)
Smooth readout installed (A3 reads out through exp42_sim_adapter); B1 is
classical. The runner's isfinite(grad) assert is kept; non-finite recorded.
The runner's verdict key is `rung_learnable_L2r`; the registered alias
`rung_learnable_L2_r` is added in the provenance block.

Usage:
  python r_exp43b_p1_L2.py
  python r_exp43b_p1_L2.py --smoke   # lr 0.02, b8, seed 0, 2 epochs -> r_smoke/
"""

import argparse
import json
import os
import sys
import time

import r_common as rc

PATCH = rc.patch_readout()
import exp43b_phase1 as m                                    # noqa: E402

ORIG_TRAIN = m.train_run
ORIG_SRC_SHA = {"exp43b_phase1.py": rc.file_sha256("exp43b_phase1.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = "r_smoke" if args.smoke else "."
    os.makedirs(outdir, exist_ok=True)
    rc.rule3(os.path.join(outdir, "results_exp43b_p1_L2r.json"))
    state = {"results_path": None, "nonfinite": []}

    def load_rung(rung):
        assert rung == "L2r", rung
        return rc.load_pinned(rc.L2R_DATA, rc.L2R_SHA256)

    fresh = rc.make_fresh_results_path(
        {"results_exp43b_phase1": "results_exp43b_p1_L2r"}, outdir)

    def fresh_results_path(base):
        state["results_path"] = fresh(base)
        return state["results_path"]

    def train_run(rung, arm, lr, batch, seed, data, verbs):
        rc.assert_patched()
        try:
            return ORIG_TRAIN(rung, arm, lr, batch, seed, data, verbs)
        except AssertionError as e:
            if "non-finite" in str(e):
                rec = {"rung": rung, "arm": arm, "lr": lr, "batch": batch,
                       "seed": seed, "error": str(e), "time": time.time()}
                with open(os.path.join(m.CKPT_DIR, "NONFINITE_%s_%s_lr%g_b%d_s%d"
                                       ".json" % (rung, arm, lr, batch, seed)),
                          "w") as f:
                    json.dump(rec, f)
                state["nonfinite"].append(rec)
            raise

    m.RUNGS = ("L2r",)
    m.load_rung = load_rung
    m.train_run = train_run
    m.fresh_results_path = fresh_results_path
    m.CKPT_DIR = os.path.join(outdir, "exp43b_p1_r_ckpt")
    if args.smoke:
        m.GRID_LRS = (0.02,)
        m.GRID_BATCHES = (8,)
        m.SEEDS = (0,)
        m.MAX_EPOCHS = 2

    old = sys.argv
    sys.argv = [old[0]]
    try:
        r = m.main()
    finally:
        sys.argv = old
    p = state["results_path"]
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            out = json.load(f)
        v = out["verdicts"]
        rc.inject_provenance(p, {
            "registration": rc.REGISTRATION,
            "wrapped_runner": "exp43b_phase1.py (unedited)",
            "wrapped_source_sha256": ORIG_SRC_SHA, "smoke": args.smoke,
            "readout_patch": PATCH,
            "rung": "L2r (%s, sha %s)" % (rc.L2R_DATA, rc.L2R_SHA256),
            "nonfinite_events": state["nonfinite"],
            "verdict_alias": {"rung_learnable_L2_r": v.get("rung_learnable_L2r")},
        })
        print("[r_exp43b_p1_L2] provenance added -> %s" % p, flush=True)
    return r or 0


if __name__ == "__main__":
    sys.exit(main())
