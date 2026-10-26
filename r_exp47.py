# -*- coding: utf-8 -*-
"""r_exp47.py -- doc 25 SSR.4(c): exp47 (B5 on L2r; A4 reference = exp45b_r).
Decides `b5_learns_L2_r` and `l2_crossing_survives_capacity_control_r` with
the verbatim doc 31 definitions (F3 init audit and memorisation control as
registered, both on L2r).

WRAPS exp47_l2_capacity.py (never edited). Rebinds:
  load_rung          -> stories_exp43b_L2r.json under the pinned sha
  A4_RESULTS         -> results_exp45b_r.json           (the A4 reference)
  A4_CKPT_DIR        -> exp45b_r_ckpt
  CKPT_DIR           -> exp47_r_ckpt
  fresh_results_path -> results_exp47_r (Rule 3)
The runner's own integrity HALT is untouched and now bites against exp45b_r:
it requires results_exp45b_r.json's dataset_sha256["L2"] == the L2r sha, and
each exp45b_r_ckpt/L2_A4_* record to match results_exp45b_r's per_rung["L2"].
B5 is classical (no quantum readout); the smooth-readout patch is installed
anyway for uniformity (no-op for B5). isfinite(grad) assert kept.

Usage:
  python r_exp47.py
  python r_exp47.py --smoke     # needs r_smoke/results_exp45b_r.json; 1 cfg,
                                # 1 seed, 2 epochs, memo 2 epochs -> r_smoke/
"""

import argparse
import json
import os
import sys
import time

import r_common as rc

PATCH = rc.patch_readout()
import exp47_l2_capacity as m                               # noqa: E402

ORIG_TRAIN = m.train_run
ORIG_SRC_SHA = {"exp47_l2_capacity.py": rc.file_sha256("exp47_l2_capacity.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = "r_smoke" if args.smoke else "."
    a4_results = os.path.join(outdir, "results_exp45b_r.json")
    if not os.path.exists(a4_results):
        print("[r_exp47] HALT: %s missing -- exp45b_r must finish first"
              % a4_results)
        return 1
    rc.rule3(os.path.join(outdir, "results_exp47_r.json"))
    state = {"results_path": None, "nonfinite": []}

    def load_rung():
        items, meta, _ = rc.load_pinned(rc.L2R_DATA, rc.L2R_SHA256)
        return items, meta

    fresh = rc.make_fresh_results_path({"results_exp47": "results_exp47_r"},
                                       outdir)

    def fresh_results_path(base):
        state["results_path"] = fresh(base)
        return state["results_path"]

    def train_run(seed, lr, batch, data, verbs, max_epochs, train_items=None,
                  select_on_val=True, tag=""):
        try:
            return ORIG_TRAIN(seed, lr, batch, data, verbs, max_epochs,
                              train_items=train_items,
                              select_on_val=select_on_val, tag=tag)
        except AssertionError as e:
            if "non-finite" in str(e):
                rec = {"arm": "B5", "seed": seed, "lr": lr, "batch": batch,
                       "tag": tag, "error": str(e), "time": time.time()}
                os.makedirs(m.CKPT_DIR, exist_ok=True)
                with open(os.path.join(m.CKPT_DIR, "NONFINITE_%s_lr%s_b%d_s%d"
                                       ".json" % (tag.strip(":"), lr, batch,
                                                  seed)), "w") as f:
                    json.dump(rec, f)
                state["nonfinite"].append(rec)
            raise

    m.load_rung = load_rung
    m.fresh_results_path = fresh_results_path
    m.train_run = train_run
    m.A4_RESULTS = a4_results
    m.A4_CKPT_DIR = os.path.join(outdir, "exp45b_r_ckpt")
    m.CKPT_DIR = os.path.join(outdir, "exp47_r_ckpt")
    if args.smoke:
        m.GRID = [(0.02, 8)]
        m.N_SEEDS = 1
        m.MAX_EPOCHS = 2
        m.MEMO_EPOCHS = 2

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
            "wrapped_runner": "exp47_l2_capacity.py (unedited)",
            "wrapped_source_sha256": ORIG_SRC_SHA, "smoke": args.smoke,
            "readout_patch": PATCH,
            "rung_L2_is": "L2r (%s, sha %s)" % (rc.L2R_DATA, rc.L2R_SHA256),
            "a4_reference": {"results": m.A4_RESULTS,
                             "results_sha256": rc.file_sha256(m.A4_RESULTS),
                             "ckpt_dir": m.A4_CKPT_DIR},
            "nonfinite_events": state["nonfinite"],
            "verdict_alias": {
                "b5_learns_L2_r": v.get("b5_learns_L2"),
                "l2_crossing_survives_capacity_control_r":
                    v.get("l2_crossing_survives_capacity_control")},
        })
        print("[r_exp47] provenance added -> %s" % p, flush=True)
    return r


if __name__ == "__main__":
    sys.exit(main())
