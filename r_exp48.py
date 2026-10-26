# -*- coding: utf-8 -*-
"""r_exp48.py -- doc 25 SSR.4(d): exp48 on L2r (CONDITIONAL). Decides
`tier_m_r` and `content_earned_A4_r` with the verbatim doc 32 definitions.

REFUSES TO RUN unless results_exp45b_r.json carries
rung_learnable_L2_2q = true (a real boolean TRUE; "UNINTERPRETABLE"/
"UNDECIDABLE"/false all refuse). SSR.5: on FALSE, exp48_r is skipped.

A4_CONFIG_r re-resolution rule (doc 25 SSR.4(d), fixed before data):
  the (lr, batch) with the highest MEAN best-val across the two exp45b_r L2
  seeds; ties -> lower lr, then smaller batch. Implemented in resolve_a4().
  (Mean best-val = mean of the per-run `val_acc`, which exp45b records at the
  best-val epoch. Equality is exact on these k/300 fractions, compared after
  rounding to 12 d.p.)

WRAPS exp48_l2_phase2.py (never edited). Rebinds:
  RUNG       "L2" -> "L2r": the runner's inline open("stories_exp43b_%s.json"
             % RUNG) then reads stories_exp43b_L2r.json; ckpt keys become
             L2r_*; the output's "rung" field reads "L2r". The pinned sha is
             asserted before main() runs.
  A4_CONFIG  -> A4_CONFIG_r (resolved above)
  REUSE      -> A4 seeds 0-1 from exp45b_r_ckpt/L2_A4_lr{lr}_b{b}_s{0,1};
             the runner re-evaluates them by forward pass and requires the
             recomputed test accuracy to equal exp45b_r's (Rule 6).
  CKPT_DIR   -> exp48_r_ckpt ; fresh_results_path -> results_exp48_r (Rule 3)
  A5_CONFIG  unchanged (0.02, 8): the runner's frozen constant (see FLAGS).
Smooth readout installed; isfinite(grad) assert kept; non-finite recorded.

Usage:
  python r_exp48.py
  python r_exp48.py --smoke   # r_smoke/, 1 seed, 1 B5 cfg, 2 epochs, C1 cap 20
                              # (the conditional refusal is OVERRIDDEN and
                              # recorded as such in smoke only)
"""

import argparse
import json
import os
import sys
import time

import r_common as rc

PATCH = rc.patch_readout()
import exp48_l2_phase2 as m                                  # noqa: E402

ORIG_TRAIN = m.train_run
ORIG_C1 = m.c1_exhaustive_2q
ORIG_SRC_SHA = {"exp48_l2_phase2.py": rc.file_sha256("exp48_l2_phase2.py")}


def resolve_a4(res):
    """doc 25 SSR.4(d): highest mean best-val over the two exp45b_r L2 seeds;
    ties -> lower lr -> smaller batch."""
    by = {}
    for r in res["per_rung"]["L2"]:
        by.setdefault((r["lr"], r["batch"]), []).append((r["seed"], r["val_acc"]))
    table = []
    for (lr, b), runs in by.items():
        seeds = sorted(s for s, _ in runs)
        table.append({"lr": lr, "batch": b, "seeds": seeds,
                      "mean_best_val": sum(v for _, v in runs) / len(runs)})
    table.sort(key=lambda t: (-round(t["mean_best_val"], 12), t["lr"], t["batch"]))
    top = table[0]
    tie = [t for t in table
           if round(t["mean_best_val"], 12) == round(top["mean_best_val"], 12)]
    return (top["lr"], top["batch"]), {"rule": "doc 25 SSR.4(d)",
                                       "table": table, "tied_at_top": len(tie),
                                       "selected": {"lr": top["lr"],
                                                    "batch": top["batch"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = "r_smoke" if args.smoke else "."
    src_res = os.path.join(outdir, "results_exp45b_r.json")
    if not os.path.exists(src_res):
        print("[r_exp48] HALT: %s missing" % src_res)
        return 1
    with open(src_res, encoding="utf-8") as f:
        res = json.load(f)
    gate = res["verdicts"].get("rung_learnable_L2_2q")
    override = False
    if gate is not True:
        if not args.smoke:
            print("[r_exp48] REFUSED: rung_learnable_L2_2q_r = %r (doc 25 SSR.5: "
                  "exp48_r runs only on TRUE)" % (gate,))
            return 2
        override = True
        print("[r_exp48] SMOKE ONLY: conditional refusal overridden "
              "(rung_learnable_L2_2q_r=%r in the smoke source)" % (gate,))
    if not args.smoke:
        assert res["dataset_sha256"]["L2"] == rc.L2R_SHA256
        n_per = {}
        for r in res["per_rung"]["L2"]:
            n_per[(r["lr"], r["batch"])] = n_per.get((r["lr"], r["batch"]), 0) + 1
        assert all(n == 2 for n in n_per.values()), \
            "SSR.4(d) needs exactly two exp45b_r seeds per config: %s" % n_per
    rc.load_pinned(rc.L2R_DATA, rc.L2R_SHA256)
    rc.rule3(os.path.join(outdir, "results_exp48_r.json"))

    a4_cfg, resolution = resolve_a4(res)
    ck45 = os.path.join(outdir, "exp45b_r_ckpt")
    reuse = {("A4", s): os.path.join(ck45, "L2_A4_lr%s_b%d_s%d"
                                     % (a4_cfg[0], a4_cfg[1], s))
             for s in (0, 1)}
    print("[r_exp48] A4_CONFIG_r = lr %s batch %d (%s)"
          % (a4_cfg[0], a4_cfg[1], resolution["table"]), flush=True)

    state = {"results_path": None, "nonfinite": []}
    fresh = rc.make_fresh_results_path({"results_exp48": "results_exp48_r"},
                                       outdir)

    def fresh_results_path(base):
        state["results_path"] = fresh(base)
        return state["results_path"]

    def train_run(arm, seed, lr, batch, data, verbs, max_epochs, tag=""):
        rc.assert_patched()
        try:
            return ORIG_TRAIN(arm, seed, lr, batch, data, verbs, max_epochs,
                              tag=tag)
        except AssertionError as e:
            if "non-finite" in str(e):
                rec = {"arm": arm, "seed": seed, "lr": lr, "batch": batch,
                       "error": str(e), "time": time.time()}
                os.makedirs(m.CKPT_DIR, exist_ok=True)
                with open(os.path.join(m.CKPT_DIR, "NONFINITE_%s_lr%s_b%d_s%d"
                                       ".json" % (arm, lr, batch, seed)),
                          "w") as f:
                    json.dump(rec, f)
                state["nonfinite"].append(rec)
            raise

    m.RUNG = "L2r"
    m.A4_CONFIG = a4_cfg
    m.REUSE = reuse
    m.CKPT_DIR = os.path.join(outdir, "exp48_r_ckpt")
    m.train_run = train_run
    m.fresh_results_path = fresh_results_path
    if args.smoke:
        m.N_SEEDS = 1
        m.MAX_EPOCHS = 2
        m.MIN_STEPS = 0
        m.B5_GRID = [(0.02, 8)]
        m.c1_exhaustive_2q = lambda models, compiled, verbs, cap=None: \
            ORIG_C1(models, compiled, verbs, cap=20)

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
            "wrapped_runner": "exp48_l2_phase2.py (unedited)",
            "wrapped_source_sha256": ORIG_SRC_SHA, "smoke": args.smoke,
            "smoke_conditional_override": override,
            "readout_patch": PATCH,
            "rung_L2_is": "L2r (%s, sha %s)" % (rc.L2R_DATA, rc.L2R_SHA256),
            "a4_config_r_resolution": resolution,
            "a4_reuse_source": {"results": src_res,
                                "results_sha256": rc.file_sha256(src_res),
                                "reuse": {"%s_s%d" % k: v2
                                          for k, v2 in reuse.items()}},
            "a5_config_note": "A5_CONFIG kept at the runner's frozen (0.02, 8)",
            "nonfinite_events": state["nonfinite"],
            "verdict_alias": {"tier_m_r": v.get("tier_m_passed"),
                              "content_earned_A4_r": v.get("content_earned_A4")},
        })
        print("[r_exp48] provenance added -> %s" % p, flush=True)
    return r


if __name__ == "__main__":
    sys.exit(main())
