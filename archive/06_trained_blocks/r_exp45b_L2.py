# -*- coding: utf-8 -*-
"""r_exp45b_L2.py -- doc 25 SSR.4(b): exp45b re-run, L2r ARM ONLY, with the
registered L1 reuse. Decides `rung_learnable_L2_2q_r`.

WRAPS exp45b_phase1.py (never edited): imports it, rebinds, calls its main().

What is rebound, and why each is faithful:
  load_rung        "L2" -> stories_exp43b_L2r.json (pinned sha); "L1" -> the
                   UNCHANGED stories_exp43b_L1.json (pinned sha 88cd70ad..).
                   The runner's own sha is read from meta; both are re-verified
                   against the canonical recomputation here.
  CKPT_DIR         exp45b_r_ckpt (new). The runner's L1/L2 loop is a literal
                   ("L1", "L2") tuple and cannot be rebound, so L1 is restricted
                   OUT through the runner's OWN checkpoint-reuse path: the 8
                   committed exp45b_ckpt/L1_A4_* records are copied in (shas
                   recorded, each cross-checked against results_exp45b_phase1)
                   and the runner prints "[ckpt] ... loaded" for them. L1 is
                   never retrained. A guard makes any L1 training call raise.
  verify_sentinel  REUSED from results_exp45b_phase1.json (doc 25 SSR.4(b):
                   "the sentinel and the memorisation controls stand from
                   exp45b"). They are L1-based; re-running them = re-running L1.
  train_run        memo: calls (tag "memo:") return the RECORDED exp45b
                   memorisation values for the same (arm, lr, batch, seed);
                   L1: calls raise; L2: calls go to the original train_run.
  fresh_results_path  results_exp45b_phase1 -> results_exp45b_r (Rule 3:
                   refuses to overwrite; no suffixing).
  readout          smooth |z|^2 (r_common.patch_readout); the runner's
                   isfinite(grad) assert is KEPT; a non-finite is recorded in
                   <ckpt>/NONFINITE_*.json and re-raised (the run halts).

After main() writes its results file, an `r_wrapper` provenance block is added
to it, including the registered boolean alias `rung_learnable_L2_2q_r`.

Usage:
  python r_exp45b_L2.py                      # the registered run
  python r_exp45b_L2.py --smoke              # 1 cfg x 1 seed x 2 epochs -> r_smoke/
"""

import argparse
import json
import os
import shutil
import sys
import time

import r_common as rc

PATCH = rc.patch_readout()
import exp45b_phase1 as m                                   # noqa: E402

ORIG = {"train_run": m.train_run, "load_rung": m.load_rung,
        "verify_sentinel": m.verify_sentinel}
ORIG_SRC_SHA = {"exp45b_phase1.py": rc.file_sha256("exp45b_phase1.py"),
                "exp45_model.py": rc.file_sha256("exp45_model.py")}


def _load_orig_results():
    with open(rc.EXP45B_ORIG_RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    assert res["dataset_sha256"]["L1"] == rc.L1_SHA256, "exp45b L1 sha drift"
    assert res["verdicts"]["a4_learns_L1"] is True
    return res


def install(ckpt_dir, outdir, smoke):
    orig = _load_orig_results()
    state = {"results_path": None, "nonfinite": []}

    def load_rung(rung):
        if rung == "L1":
            items, meta, _ = rc.load_pinned(rc.L1_DATA, rc.L1_SHA256)
        elif rung == "L2":
            items, meta, _ = rc.load_pinned(rc.L2R_DATA, rc.L2R_SHA256)
        else:
            raise ValueError(rung)
        return items, meta

    def verify_sentinel(*_a, **_k):
        s = dict(orig["sentinel"])
        s["r_reused_from"] = rc.EXP45B_ORIG_RESULTS
        s["r_reuse_basis"] = ("doc 25 SSR.4(b): L1 reuse registered; the "
                              "sentinel stands from exp45b (computed with the "
                              "abs()**2 readout, on the unchanged L1 data)")
        return s

    memo_rec = {(arm, r["lr"], r["batch"], r["seed"]): r["best_train_acc"]
                for arm, blk in orig["memorisation_controls"].items()
                for r in blk["runs"]}

    def train_run(arm, seed, lr, batch, data, verbs, max_epochs,
                  train_items=None, select_on_val=True, tag=""):
        if tag == "memo:":
            key = (arm, lr, batch, seed)
            if key not in memo_rec:
                raise RuntimeError("no recorded exp45b memorisation run %s"
                                   % (key,))
            return None, {"train_acc": memo_rec[key],
                          "r_reused_from": rc.EXP45B_ORIG_RESULTS}
        if tag != "L2:":
            raise RuntimeError("r_exp45b_L2: refusing to TRAIN %r (L1 is "
                               "reused from exp45b_ckpt, never retrained)" % tag)
        rc.assert_patched()
        try:
            return ORIG["train_run"](arm, seed, lr, batch, data, verbs,
                                     max_epochs, train_items=train_items,
                                     select_on_val=select_on_val, tag=tag)
        except AssertionError as e:
            if "non-finite" in str(e):
                rec = {"arm": arm, "seed": seed, "lr": lr, "batch": batch,
                       "error": str(e), "time": time.time(),
                       "policy": "recorded, never skipped; run halts"}
                p = os.path.join(ckpt_dir, "NONFINITE_%s_lr%s_b%d_s%d.json"
                                 % (arm, lr, batch, seed))
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(rec, f)
                state["nonfinite"].append(rec)
            raise

    fresh = rc.make_fresh_results_path(
        {"results_exp45b_phase1": "results_exp45b_r",
         "results_exp45b_HALTED": "results_exp45b_r_HALTED"}, outdir)

    def fresh_results_path(base):
        p = fresh(base)
        state["results_path"] = p
        return p

    m.load_rung = load_rung
    m.verify_sentinel = verify_sentinel
    m.train_run = train_run
    m.fresh_results_path = fresh_results_path
    m.CKPT_DIR = ckpt_dir
    if smoke:
        m.GRID = [(0.02, 8)]
        m.N_SEEDS = 1
        m.MAX_EPOCHS = 2

    # ---- L1 reuse through the runner's own ckpt path
    os.makedirs(ckpt_dir, exist_ok=True)
    l1_ref = {(r["lr"], r["batch"], r["seed"]): r
              for r in orig["per_rung"]["L1"]}
    copies = []
    for lr, b in m.GRID:
        for s in range(m.N_SEEDS):
            key = "L1_A4_lr%s_b%d_s%d" % (lr, b, s)
            ent = {"key": key}
            for ext in (".json", ".pt"):
                src = os.path.join(rc.EXP45B_ORIG_CKPT, key + ext)
                dst = os.path.join(ckpt_dir, key + ext)
                if os.path.exists(dst):
                    assert rc.file_sha256(dst) == rc.file_sha256(src), dst
                else:
                    shutil.copy2(src, dst)
                ent["sha256" + ext] = rc.file_sha256(dst)
            with open(os.path.join(ckpt_dir, key + ".json"), encoding="utf-8") as f:
                d = json.load(f)
            ref = l1_ref[(lr, b, s)]
            ent["matches_results_exp45b_phase1"] = bool(
                d["train_acc"] == ref["train_acc"] and d["val_acc"] == ref["val_acc"]
                and d["gate_met_at_bestval"] == ref["gate_met_at_bestval"])
            assert ent["matches_results_exp45b_phase1"], key
            copies.append(ent)
    return state, copies, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = "r_smoke" if args.smoke else "."
    ckpt_dir = os.path.join(outdir, "exp45b_r_ckpt")
    if not args.smoke:
        rc.rule3("results_exp45b_r.json")
    else:
        os.makedirs(outdir, exist_ok=True)
        rc.rule3(os.path.join(outdir, "results_exp45b_r.json"))
    state, copies, orig = install(ckpt_dir, outdir, args.smoke)

    old_argv = sys.argv
    sys.argv = [old_argv[0]]              # never pass --smoke to the runner
    try:
        rc_ = m.main()
    finally:
        sys.argv = old_argv
    p = state["results_path"]
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            out = json.load(f)
        v = out.get("verdicts", {})
        rc.inject_provenance(p, {
            "registration": rc.REGISTRATION,
            "wrapped_runner": "exp45b_phase1.py (unedited)",
            "wrapped_source_sha256": ORIG_SRC_SHA,
            "smoke": args.smoke,
            "readout_patch": PATCH,
            "rung_L2_is": "L2r (%s, sha %s)" % (rc.L2R_DATA, rc.L2R_SHA256),
            "L1_reuse": {
                "basis": "doc 25 SSR.4(b): L1 data unchanged (sha 88cd70ad..), "
                         "a4_learns_L1 = TRUE, sentinel and memorisation "
                         "controls stand from exp45b",
                "L1_phase1_ckpt_copies": copies,
                "sentinel": "reused from results_exp45b_phase1.json",
                "memorisation_controls": "reused from results_exp45b_phase1."
                                         "json (recorded best_train_acc per "
                                         "arm/lr/batch/seed)",
                "note": "these were computed with the abs()**2 readout; the "
                        "forward maths is identical (anchor delta in "
                        "readout_patch)"},
            "literal_field_caveat": "`datasets_reused_verbatim: true` is a "
                                    "literal of the wrapped runner; here L2 is "
                                    "the REGENERATED L2r and only L1 is reused",
            "nonfinite_events": state["nonfinite"],
            "verdict_alias": {"rung_learnable_L2_2q_r":
                              v.get("rung_learnable_L2_2q")},
        })
        print("[r_exp45b_L2] provenance added -> %s ; rung_learnable_L2_2q_r=%s"
              % (p, v.get("rung_learnable_L2_2q")), flush=True)
    return rc_


if __name__ == "__main__":
    sys.exit(main())
