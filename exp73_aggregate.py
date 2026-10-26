# -*- coding: utf-8 -*-
"""exp73_aggregate.py -- doc 61's ONLY results writer (doc 53 SS7.3).

Ships in the SAME COMMIT as `exp73_run.py` and `exp73_comparator.py` (doc 61 SS16.5),
and is written to be run COLD: it reads the per-run checkpoint JSONs in `exp73_ckpt/`
and `exp73_ckpt/comparator.json`, and needs nothing that is not on disk.

THE GATING ORDER IS REGISTERED AND IS ENFORCED HERE (doc 61 SS15.4 / SS17.3 item 11)

    anchor -> floors -> `binds` per cell -> `amplitude_helps/hurts`
           -> `init_helps/hurts` -> `cc` and SS10.5's ladder placement
           -> SS9's descriptives

  * **doc 61 SS10.1 `anchor_replicates_committed` IS READ FIRST AND IS BLOCKING.**
    If it is FALSE the exp73 pipeline has drifted, every amplitude number is a
    comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS READ: this
    script writes the anchor file, writes `results_exp73.json` with every verdict
    `null` and `run_readable = false`, and stops. Hunt the build; interpret nothing.
  * **A TRIGGERED CELL WHOSE FLOOR HAS NOT BEEN RUN HAS NO VERDICT AND ITS BOOLEANS
    ARE STORED AS `null`, NEVER AS `false`** (SS15.4). The same applies to any cell
    whose runs are incomplete, and to A4 while it is unrun.

WHAT IS AND IS NOT A CRITERION
  REGISTERED BOOLEANS (SS10):  `anchor_replicates_committed`; `binds_<cell>` per
    cell; `amplitude_helps_T{1,2}` / `amplitude_hurts_T{1,2}`;
    `init_helps_<rung>_T{t}` / `init_hurts_<rung>_T{t}`.
  INERT BY CONSTRUCTION, COMPUTED AND STORED, NO READING DEPENDS ON THEM:
    `amplitude_helps_T1` (its bar is 1.0420 and accuracy is bounded by 1 -- the
    T = 1 cell is ONE-SIDED BY CONSTRUCTION, SS10.3) and `bug_hunt_triggered` (both
    bounds are 1.0000 by proof, so bound + 0.06 > 1 and NO ACHIEVABLE ACCURACY CAN
    TRIGGER IT, SS12.1).
  DESCRIPTIVE, NO BOOLEAN: `cc` and SS10.5's ladder placement; SS9's purity traces;
    SS10.6's reported-not-registered list. **NONE MAY BE PROMOTED ONTO A CRITERION
    AFTER RESULTS EXIST** (the exp44 no-promotion clause, doc 55 SS18.2).
  DELIBERATELY NOT REGISTERED (SS10.7): any criterion reading a bound; any T1-T2
    subtraction as a criterion; ANY A3-vs-A4 BOOLEAN; any advantage-shaped quantity;
    any comparison against exp72's rescue arms.

OUTPUTS (doc 61 SS16.5's registered filenames; Rule 3 -- nothing is overwritten)
    results_exp73_anchors.json          the SS5.3 gate, separate by Rule 3
    results_exp73.json                  the run
    results_exp73_floors_stage2.json    the procedure-matched floors, separate
                                        because they may be run later than the arms

    python exp73_comparator.py     # FIRST -- the arbiter tiers (i)/(ii)/(iii)
    python exp73_aggregate.py
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

import exp73_run as R

SPEC = R.SPEC
RESULTS_OUT = "results_exp73.json"
FLOORS_OUT = "results_exp73_floors_stage2.json"
COMPARATOR = os.path.join(R.CKPT_DIR, "comparator.json")

BIND_BAR = 0.65                # doc 54 SS7's `anchor_real_category_binds`, unchanged
FLOOR_MULTIPLE = 2.0           # doc 61 SS10.2
DELTA_BAR = 0.06               # 2 x the binomial SD at n_test = 300 (sqrt(.25/300))
A4_INIT_TRIGGER = 0.06         # orchestrator ruling 2

# doc 56 SS18.1, verbatim -- the committed composition-fragility ladder.
COMMITTED_LADDER = [("angle (FA / EA-ctrl)", 0.004),
                    ("CZ-across (N-across)", 0.032),
                    ("eps-register (P-eps)", 0.132),
                    ("old-map register (F2 / A0-W)", 0.253),
                    ("entangled register (N-in)", 0.310)]


def sd(xs):
    xs = list(xs)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def load_runs(ckpt_dir, tag=""):
    """Every per-run checkpoint JSON in `ckpt_dir`, keyed (arm, seed)."""
    out = {}
    if not os.path.isdir(ckpt_dir):
        return out
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.endswith(".json") or fn == "comparator.json":
            continue
        try:
            r = json.load(open(os.path.join(ckpt_dir, fn), encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(r, dict) or "arm" not in r or "seed" not in r:
            continue
        if r.get("tag", "") != tag:
            continue
        out[(r["arm"], int(r["seed"]))] = r
    return out


def arm_block(runs, arm, n_seeds):
    """None when the cell is INCOMPLETE -- an incomplete cell has NO verdict."""
    rows = [runs[(arm, s)] for s in range(n_seeds) if (arm, s) in runs]
    if len(rows) < n_seeds:
        return {"complete": False, "n_runs": len(rows),
                "seeds_present": sorted(r["seed"] for r in rows),
                "n_seeds_registered": n_seeds,
                "note": ("INCOMPLETE -- doc 61 SS15.4: a cell without its full seed "
                         "set has NO verdict and its booleans are stored as null, "
                         "never as false")}
    rows.sort(key=lambda r: r["seed"])
    acc = [r["heldout_test_acc"] for r in rows]
    cfg = R.ARM_CFG[R.base_arm(arm)]
    blk = {
        "complete": True, "arm": arm, "base_arm": R.base_arm(arm),
        "is_floor": bool(arm.startswith("S-shuffle")),
        "rung": cfg["rung"], "T": cfg["T"], "init": cfg["init"],
        "encoding": cfg["enc"], "role": cfg["role"],
        "mean": float(np.mean(acc)), "sd": sd(acc), "per_seed": acc,
        "seeds": [r["seed"] for r in rows], "n_seeds": len(rows),
        "val_per_seed": [r["val_acc"] for r in rows],
        "train_per_seed": [r["train_acc"] for r in rows],
        "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
        "val_minus_heldout_per_seed": [r["val_minus_heldout"] for r in rows],
        "mean_val_minus_heldout": float(np.mean(
            [r["val_minus_heldout"] for r in rows])),
        "gate_met_per_seed": [r["gate_met"] for r in rows],
        "n_params": rows[0]["n_params"], "wires": rows[0]["wires"],
        "qubits_per_referent": rows[0]["qubits_per_referent"],
        "d_truncated": rows[0]["d_truncated"],
        "init_scheme": rows[0]["init_scheme"].get("scheme"),
        "dataset_sha256": rows[0]["dataset_sha256"],
        "ckpt_sha256": [r["ckpt_sha256"] for r in rows],
        "wall_sec_total": float(sum(r["wall_sec"] for r in rows)),
    }
    blk.update(descriptives(rows))
    return blk


def descriptives(rows):
    """doc 61 SS10.6 -- REPORTED, NOT REGISTERED, and NONE MAY BE PROMOTED ONTO A
    CRITERION AFTER THE FACT (Rule 4's second sentence, the exp44 lesson)."""
    by_q, by_order, by_verb = {}, {}, {}
    for r in rows:
        for it in r["heldout_items"]:
            for cell, key in ((by_q, "q%d" % it["query_index"]),
                              (by_order, "/".join(it["order_flags"])),
                              (by_verb, it["queried_verb"])):
                cell.setdefault(key, [0, 0])
                cell[key][0] += int(it["correct"])
                cell[key][1] += 1

    def fmt(cell):
        return {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(cell.items())}

    return {"per_q_accuracy": fmt(by_q),
            "per_surface_order_accuracy": fmt(by_order),
            "per_queried_verb_accuracy": fmt(by_verb),
            "purity_traces_doc61_SS9_3": purity_summary(rows),
            "reported_not_registered_doc61_SS10_6": (
                "per-q held-out accuracy; per-surface-order accuracy; the "
                "val-minus-held-out selection gap per seed (THE LIVE OVERFITTING "
                "EXPOSURE at 408 parameters against 528 items, doc 61 SS6.5 / SS19 "
                "row 2 -- comparison points: exp69 F2 gap 0.0353, FA 0.0813); "
                "selected epoch per seed; per-verb accuracy; SS9's two purity "
                "traces; per-seed deltas. NO VERDICT READS ANY OF THESE AND NONE MAY "
                "BE PROMOTED ONTO A CRITERION AFTER RESULTS EXIST.")}


def purity_summary(rows):
    """doc 61 SS9.3 / SS9.4 -- the two traces, stratified by q and by correctness.
    DESCRIPTIVE. No criterion reads them (SS9.5)."""
    acc = {}
    for r in rows:
        for it in r["heldout_items"]:
            pur = it.get("purity")
            if not pur:
                continue
            key = ("q%d" % it["query_index"], "correct" if it["correct"] else "wrong")
            for stage, sn in pur.items():
                for kind in ("wire", "register"):
                    acc.setdefault((key, stage, kind), []).extend(sn[kind])
    out = {}
    for (key, stage, kind), vals in acc.items():
        out.setdefault("%s/%s" % key, {}).setdefault(stage, {})[kind] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n": len(vals)}
    return {
        "by_q_and_correctness": out,
        "reading_doc61_SS9_2": (
            "the per-WIRE trace is read against doc 61 SS9.2's PRE-COMPUTED "
            "INVENTORY as its baseline, NOT against 1: a generic unit vector in "
            "R^{2^q} is not a product of q single-qubit states, so the intro per-wire "
            "purity is below 1 BY CONSTRUCTION and is a DATA-DETERMINED CONSTANT OF "
            "THE ENCODING, not a circuit effect. A3 min/median/max 0.5036/0.6660/"
            "0.9693; A4 0.5046/0.5299/0.7924; NO NOUN IS WITHIN 0.05 OF A PRODUCT "
            "STATE at either rung. Here the mechanism is MAXIMAL AND UNAVOIDABLE, "
            "and it is the ENCODING, not a gate."),
        "registered_question_doc61_SS9_4": (
            "does the readout restore the per-REGISTER purity to its INTRO value "
            "(1.0, by construction) on correct q = 2 items and fail to on q = 1 "
            "items -- the q-asymmetry undo signature (doc 55 SS20.5) transposed from "
            "wires to registers? DESCRIPTIVE, NO BOOLEAN, NO CRITERION. The healthy "
            "committed profile is the wire-level one (R2-A: 0.8755 vs 0.8751 on "
            "correct q = 2; 0.8017 vs 0.8953 on q = 1)."),
        "no_promotion_doc61_SS9_5": (
            "these traces MAY describe how an arm succeeds or fails. They MAY NOT be "
            "promoted onto a criterion after results exist (the exp44 no-promotion "
            "clause), and NO VERDICT READS THEM."),
    }


def floor_clearance(arm_blk, floor_blk):
    """doc 61 SS10.2 with doc 55 SS6.6's three SD readings. THE FLOOR'S OWN SD IS
    PRIMARY; all three are computed and reported and DISAGREEMENT IS REPORTED AS
    DISAGREEMENT RATHER THAN RESOLVED SILENTLY."""
    if not (arm_blk.get("complete") and floor_blk and floor_blk.get("complete")):
        return None
    a, f = arm_blk["mean"], floor_blk["mean"]
    s_floor, s_arm = floor_blk["sd"], arm_blk["sd"]
    s_pool = float(np.sqrt((s_floor ** 2 + s_arm ** 2) / 2.0))
    read = {}
    for name, s in (("floor_sd_PRIMARY", s_floor), ("arm_sd", s_arm),
                    ("pooled_sd", s_pool)):
        read[name] = {"sd": s, "bar": f + FLOOR_MULTIPLE * s,
                      "clears": bool(a >= f + FLOOR_MULTIPLE * s),
                      "margin_in_sd": (float((a - f) / s) if s > 0 else None)}
    agree = len({read[k]["clears"] for k in read}) == 1
    # doc 61 SS15.5 -- THE CONTAMINATION BRANCH, live at these parameter counts.
    contaminated = bool(f >= BIND_BAR or (a - f) < DELTA_BAR)
    return {
        "arm_mean": a, "floor_mean": f, "delta": a - f,
        "readings": read, "primary": "floor_sd_PRIMARY",
        "clears_floor": read["floor_sd_PRIMARY"]["clears"],
        "readings_agree": bool(agree),
        "disagreement_note": (None if agree else
                              "doc 55 SS6.6: the three SD readings DISAGREE and that "
                              "is REPORTED AS DISAGREEMENT rather than resolved "
                              "silently. The floor's own SD is primary."),
        "floor_contaminated_doc61_SS15_5": contaminated,
        "contamination_rule": (
            "doc 61 SS15.5: if a floor RISES TO MEET ITS ARM the cell is reported as "
            "floor-contaminated with BOTH numbers and IS NOT READ. Operationalised "
            "here, transparently, as floor_mean >= %.2f (the floor itself reaches the "
            "binding bar) OR arm_mean - floor_mean < %.2f (the separation is inside "
            "the registered resolution). At 408 parameters against 528 items this is "
            "A LIVE RISK AND NOT A FORMALITY." % (BIND_BAR, DELTA_BAR)),
        "floor_construction": (
            "doc 61 SS15.1: the committed S-shuffle (`exp69_run.relabel_shuffle_q`, "
            "imported unchanged) -- bearer designation reassigned PER PAIR, "
            "independently of the features -- same items, same splits, same 5 seeds, "
            "run under the IDENTICAL ARM PROCEDURE. **THE INIT IS PART OF THE "
            "PROCEDURE, so each cell carries its OWN floor** (doc 58 SS14's argument, "
            "inherited: a floor measured under one optimisation procedure does not "
            "bound what another can achieve on severed labels)."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=RESULTS_OUT)
    ap.add_argument("--floors-out", default=FLOORS_OUT)
    ap.add_argument("--anchor-out", default=R.ANCHOR_OUT)
    ap.add_argument("--comparator", default=None)
    a = ap.parse_args()
    t0 = time.time()
    R.CKPT_DIR = a.ckpt_dir
    comp_path = a.comparator or os.path.join(a.ckpt_dir, "comparator.json")

    store = R.load_part()
    runs = load_runs(a.ckpt_dir, tag=a.tag)
    print("[exp73-agg] %d per-run checkpoints in %s" % (len(runs), a.ckpt_dir))

    head = {
        "experiment": "exp73 -- THE AMPLITUDE RUNG (doc 61, D1 of doc 59's plan)",
        "spec": SPEC,
        "question": ("can a referent's whole (truncated, normalised) AraVec vector be "
                     "written onto its register as AMPLITUDES rather than compressed "
                     "into one or two angles -- and if so, what does the readout "
                     "actually recover from it? (doc 61 SS0.1)"),
        "orchestrator_rulings": {
            "1_verb_block": ("MONOLITHIC Circuit4(2q, 3): 408 (A3) / 552 (A4) "
                             "parameters, reproducing the committed block at BOTH "
                             "q = 1 and q = 2. The brickwork is NOTED, NOT RUN and "
                             "is not implemented anywhere in the exp73 code."),
            "2_compute": ("A4 IS A CONDITIONAL SECOND STAGE -- it runs only if A3 "
                          "binds at T = 1 or shows an init effect >= 0.06. Stage 1 "
                          "is A3-only."),
            "3_batch_hoist": ("DECLINED -- no registration touches shared committed "
                              "code. doc 61 SS16.3 is not implemented."),
            "4_directional_call": "amplitude_hurts_T1 = TRUE (bar <= 0.9220), KEPT",
            "5_tier_iii_grid": ("C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2}, FROZEN, "
                                "selected on val, reported with SS8.3's restriction "
                                "qualifier")},
        "aggregated_from": a.ckpt_dir, "aggregator": os.path.basename(__file__),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "dataset": store[2], "dataset_sha256": store[1]["dataset_sha256"],
        "smoke_results": R.SMOKE_OUT, "anchor_results": a.anchor_out,
        "n_seeds": a.seeds, "sd_convention": "sample sd, ddof=1",
        "training": {"lr": R.LR, "batch": R.BATCH, "max_epochs": R.MAX_EPOCHS,
                     "rng_base": R.RNG_BASE,
                     "selection": "in-distribution val accuracy",
                     "heldout": "read ONCE from the selected checkpoint"},
        "advantage_prohibition": (
            "NO EXP73 NUMBER MAY APPEAR IN A SENTENCE CONTAINING 'ADVANTAGE' "
            "(doc 52 SS1; doc 56 SS0.1; doc 58 SS12; doc 61 SS0.3 / SS13). doc 61 "
            "SS0.3 applies doc 52 SS1's prohibition in its STRONGEST form to date: "
            "for the amplitude map the matched classical comparator is not even a "
            "CONSTRUCTION -- it is THE SAME OBJECT IN ANOTHER BASIS, and the "
            "equivalence is already in the literature (SS4.1)."),
    }

    # ------------------------------------------------------------------ STEP 1
    # doc 61 SS10.1 / SS17.3 item 8 -- THE ANCHOR, READ FIRST AND BLOCKING.
    anchor = None
    anchor_err = None
    try:
        anchor = R.anchor_report(store)
        json.dump(anchor, open(a.anchor_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[exp73-agg] wrote %s -- anchor_replicates_committed=%s "
              "(bit_exact=%s, max|d|=%.2e)"
              % (a.anchor_out, anchor["anchor_replicates_committed"],
                 anchor["bit_exact"], anchor["max_abs_per_seed_delta"]))
    except (RuntimeError, AssertionError, KeyError) as exc:
        anchor_err = str(exc)
        print("[exp73-agg] ANCHOR NOT AVAILABLE: %s" % anchor_err)

    anchor_ok = bool(anchor and anchor["anchor_replicates_committed"])
    head["anchor_gate"] = anchor or {"available": False, "error": anchor_err}
    head["anchor_gate_note"] = (
        "doc 61 SS5.3 / SS10.1: BLOCKING, READ FIRST. `A0-W-T2`'s PER-SEED held-out "
        "accuracies must reproduce results_exp69.json :: R2-W to |delta| <= 0.002 and "
        "the arm mean to |delta| <= 0.002 of 0.7293. Both runs are deterministic "
        "given their seeds, so THE HONEST EXPECTATION IS BIT-EXACT, 5/5. IF THIS "
        "FAILS the exp73 pipeline has drifted, every amplitude number is a comparison "
        "against a moving baseline, NOTHING ELSE IN THE RUN IS READ, and SS5.3.1's "
        "floor-omission argument for the anchor is VOID along with the run.")

    # ------------------------------------------------------------------ STEP 2
    # arms and their procedure-matched floors.
    cells = R.STAGE1_TRAINED + R.STAGE2_TRAINED
    arms = {c: arm_block(runs, c, a.seeds) for c in cells}
    arms[R.ANCHOR_ARM] = arm_block(runs, R.ANCHOR_ARM, a.seeds)
    floors = {c: arm_block(runs, R.FLOOR_OF[c], a.seeds) for c in cells}
    clear = {c: floor_clearance(arms[c], floors[c]) for c in cells}

    head["arms"] = arms
    head["floors"] = floors
    head["floor_clearance"] = clear
    head["stage"] = {
        "stage1_cells": R.STAGE1_TRAINED,
        "stage2_cells_conditional": R.STAGE2_TRAINED,
        "stage1_complete": bool(all(arms[c].get("complete")
                                    for c in R.STAGE1_TRAINED)),
        "stage1_floors_complete": bool(all(floors[c].get("complete")
                                           for c in R.STAGE1_TRAINED)),
        "floors_run_up_front_note": (
            "doc 61 SS15 stages the floors behind a trigger test (SS15.3). With A4 "
            "deferred by ruling 2 the WORST CASE is four A3 cells triggering, i.e. "
            "exactly the four floors run here, so running all four up front is the "
            "CONSERVATIVE choice: no verdict can be read past a missing floor "
            "(SS15.4) and it costs nothing the worst case would not have cost. "
            "SS15.2's stage-1 orientation comparison against the COMMITTED floors is "
            "reported below and CARRIES NO VERDICT."),
    }

    # doc 61 SS15.2 -- ORIENTATION ONLY, CARRIES NO VERDICT.
    smoke = (json.load(open(R.SMOKE_OUT, encoding="utf-8"))
             if os.path.exists(R.SMOKE_OUT) else {})
    committed = smoke.get("committed_imports_doc61_SS7_5", {})
    head["orientation_doc61_SS15_2"] = {
        "committed_floors": committed.get("floors"),
        "per_cell": {c: (None if not arms[c].get("complete") else {
            "heldout_mean": arms[c]["mean"],
            "vs_committed_F2_floor": arms[c]["mean"] - 0.5073 if arms[c]["T"] == 1
            else arms[c]["mean"] - 0.5020,
            "vs_committed_P_eps_floor": arms[c]["mean"] - 0.5180 if arms[c]["T"] == 1
            else arms[c]["mean"] - 0.5093}) for c in cells},
        "note": ("doc 61 SS15.2: STAGE 1 IS ORIENTATION ONLY. This comparison against "
                 "the COMMITTED matched floors of the nearest committed rung CARRIES "
                 "NO VERDICT. Its only registered function was to decide which cells "
                 "go to stage 2; here every stage-1 cell already has its OWN "
                 "procedure-matched floor, so the verdicts below read THOSE."),
    }

    # ------------------------------------------------------------------ STEP 3
    verdicts = {}
    reasons = {}

    def block(name, why):
        verdicts[name] = None
        reasons[name] = why

    verdicts["anchor_replicates_committed"] = (
        anchor["anchor_replicates_committed"] if anchor else None)
    reasons["anchor_replicates_committed"] = (
        head["anchor_gate_note"] if anchor else
        "the anchor has not been run or its checkpoints are incomplete -- doc 61 "
        "SS15.4: stored as null, NEVER as false")

    readable = anchor_ok
    if not readable:
        for c in cells:
            block("binds_%s" % c, "NOT READ -- the drift anchor gate did not pass "
                                  "(doc 61 SS5.3): every amplitude number would be a "
                                  "comparison against a moving baseline")
        for t in (1, 2):
            block("amplitude_helps_T%d" % t, "NOT READ -- anchor gate")
            block("amplitude_hurts_T%d" % t, "NOT READ -- anchor gate")
        for rung in R.RUNGS:
            for t in (1, 2):
                block("init_helps_%s_T%d" % (rung, t), "NOT READ -- anchor gate")
                block("init_hurts_%s_T%d" % (rung, t), "NOT READ -- anchor gate")
    else:
        # ---- `binds` per cell (doc 61 SS10.2), CONJUNCTIVE ----------------
        for c in cells:
            ab, cl = arms[c], clear[c]
            if not ab.get("complete"):
                block("binds_%s" % c,
                      "the cell is INCOMPLETE (%d/%d seeds) -- doc 61 SS15.4: null, "
                      "never false" % (ab.get("n_runs", 0), a.seeds))
                continue
            if cl is None:
                block("binds_%s" % c,
                      "doc 61 SS15.4: the cell's PROCEDURE-MATCHED FLOOR has not been "
                      "run, so it has NO VERDICT and this boolean is null, NEVER "
                      "false. `binds` may not be read floor-less.")
                continue
            if cl["floor_contaminated_doc61_SS15_5"]:
                block("binds_%s" % c,
                      "doc 61 SS15.5 FLOOR-CONTAMINATED: arm %.4f against floor "
                      "%.4f. The cell is reported with BOTH numbers and IS NOT READ."
                      % (cl["arm_mean"], cl["floor_mean"]))
                continue
            verdicts["binds_%s" % c] = bool(ab["mean"] >= BIND_BAR
                                            and cl["clears_floor"])
            reasons["binds_%s" % c] = (
                "doc 61 SS10.2, CONJUNCTIVE: heldout_mean %.4f >= %.2f is %s AND "
                "heldout_mean >= floor_mean + 2 x floor_sd (%.4f) is %s. The 0.65 is "
                "a FREE-STANDING ABSOLUTE BAR (doc 54 SS7's "
                "`anchor_real_category_binds`) and the floor condition is SEPARATE. "
                "All three SD readings stored; the floor's own SD is PRIMARY."
                % (ab["mean"], BIND_BAR, ab["mean"] >= BIND_BAR,
                   cl["readings"]["floor_sd_PRIMARY"]["bar"], cl["clears_floor"]))

        # ---- `amplitude_helps/hurts` (doc 61 SS10.3) ----------------------
        best_committed = (committed.get("best_committed_2qubit_register")
                          or {"T1": 0.9820, "T2": 0.8340})
        for t in (1, 2):
            live = [c for c in cells
                    if arms[c].get("complete") and arms[c]["T"] == t
                    and clear[c] and not clear[c]["floor_contaminated_doc61_SS15_5"]]
            # doc 61 SS15.3 trigger (b) / doc 52 SS3: the BEST cell at each T must
            # carry a procedure-matched floor before a GAIN criterion is read.
            unfloored = [c for c in cells
                         if arms[c].get("complete") and arms[c]["T"] == t
                         and clear[c] is None]
            if not live or unfloored:
                why = ("no complete, floor-carrying cell at T = %d" % t if not live
                       else "cells %r at T = %d are complete but carry NO "
                            "procedure-matched floor. doc 52 SS3 binds: a GAIN "
                            "criterion needs a null exactly as much as an absolute "
                            "one, and doc 61 SS15.3 trigger (b) exists so the best "
                            "cell at each T is NEVER read floor-less."
                            % (unfloored, t))
                block("amplitude_helps_T%d" % t, why)
                block("amplitude_hurts_T%d" % t, why)
                continue
            best_cell = max(live, key=lambda c: arms[c]["mean"])
            delta = arms[best_cell]["mean"] - best_committed["T%d" % t]
            verdicts["amplitude_helps_T%d" % t] = bool(delta >= DELTA_BAR)
            verdicts["amplitude_hurts_T%d" % t] = bool(delta <= -DELTA_BAR)
            reasons["amplitude_T%d" % t] = {
                "best_amplitude_cell": best_cell,
                "best_amplitude_mean": arms[best_cell]["mean"],
                "best_committed_2qubit_mean": best_committed["T%d" % t],
                "delta": delta, "bar": DELTA_BAR,
                "helps_bar": best_committed["T%d" % t] + DELTA_BAR,
                "hurts_bar": best_committed["T%d" % t] - DELTA_BAR,
                "cells_considered": {c: arms[c]["mean"] for c in live},
                "unpaired_doc61_SS10_3_1": (
                    "F2 and P-eps are COMMITTED measurements on the same frozen "
                    "dataset, same splits, same 5 seeds, so the populations match and "
                    "doc 47 SS18.5's prohibition is not engaged -- but the arms were "
                    "not run in the same session and NO PER-SEED PAIRING IS CLAIMED. "
                    "MEANS AGAINST MEANS, sds reported, NO PAIRED TEST."),
                "both_false_is_a_result": (
                    "doc 61 SS10.3: BOTH MAY BE FALSE AT T = 2, AND THAT IS A RESULT, "
                    "NOT A FAILURE -- it reads as 'indistinguishable at this power'. "
                    "Both may not be TRUE simultaneously. doc 55 SS20.2's knife-edge "
                    "(+0.0593 against a 0.06 bar) is the standing reminder that the "
                    "region near the bar is where readings are made or lost."),
            }
        if verdicts.get("amplitude_helps_T1") is not None:
            reasons["amplitude_helps_T1_INERT"] = (
                "doc 61 SS10.3: STRUCTURALLY UNREACHABLE AND REGISTERED AS INERT. Its "
                "bar is %.4f and held-out accuracy is bounded by 1, so THE T = 1 CELL "
                "IS ONE-SIDED BY CONSTRUCTION -- only `amplitude_hurts_T1` can fire. "
                "The boolean is computed and stored for completeness and NO READING "
                "DEPENDS ON IT. **T = 2 IS THE LIVE COMPARISON**, and it is also "
                "where the ladder's action is."
                % (best_committed["T1"] + DELTA_BAR))

        # ---- `init_helps/hurts` (doc 61 SS10.4) ---------------------------
        for rung in R.RUNGS:
            for t in (1, 2):
                cu, ci = "%s-T%d-uniform" % (rung, t), "%s-T%d-identity" % (rung, t)
                if not (arms[cu].get("complete") and arms[ci].get("complete")):
                    why = ("the %s T = %d init pair is INCOMPLETE -- doc 61 SS15.4: "
                           "null, never false" % (rung, t))
                    block("init_helps_%s_T%d" % (rung, t), why)
                    block("init_hurts_%s_T%d" % (rung, t), why)
                    continue
                if not (clear[cu] and clear[ci]) or \
                        clear[cu]["floor_contaminated_doc61_SS15_5"] or \
                        clear[ci]["floor_contaminated_doc61_SS15_5"]:
                    why = ("doc 61 SS15.4: one of the two %s T = %d cells has no "
                           "procedure-matched floor or is floor-contaminated, so no "
                           "contrast is read past it" % (rung, t))
                    block("init_helps_%s_T%d" % (rung, t), why)
                    block("init_hurts_%s_T%d" % (rung, t), why)
                    continue
                dlt = arms[ci]["mean"] - arms[cu]["mean"]
                verdicts["init_helps_%s_T%d" % (rung, t)] = bool(dlt >= DELTA_BAR)
                verdicts["init_hurts_%s_T%d" % (rung, t)] = bool(dlt <= -DELTA_BAR)
                paired = [i - u for i, u in zip(arms[ci]["per_seed"],
                                                arms[cu]["per_seed"])]
                reasons["init_%s_T%d" % (rung, t)] = {
                    "identity_mean": arms[ci]["mean"],
                    "uniform_mean": arms[cu]["mean"],
                    "init_delta": dlt, "bar": DELTA_BAR,
                    "paired_by_seed_delta": paired,
                    "paired_mean": float(np.mean(paired)),
                    "paired_sd": sd(paired),
                    "note": ("doc 61 SS10.4: BOTH FALSE = indistinguishable at this "
                             "power. Paired-by-seed is AVAILABLE AND IS REPORTED; "
                             "THE BOOLEAN READS THE MEANS (doc 55's convention)."),
                    "no_direction_registered_doc61_SS11_3": (
                        "the only evidence that identity initialisation helps this "
                        "programme's registers is doc 58 SS19.1's "
                        "rescue_partial(I1, R-identity) = TRUE -- ONE INSTANCE, ONE "
                        "DRAW, on the product register at T = 2, whose confirm draw "
                        "was ordered 2026-08-27. doc 58 SS19.3 records that the "
                        "ENTANGLED register resisted identity init entirely. "
                        "Registering a direction on an unconfirmed single draw is "
                        "exactly what doc 47 SS18 exists to prevent. **doc 61 SS17.3 "
                        "item 5: the exp72 confirm on I1:R-identity must be RECORDED "
                        "IN WRITING before any exp73 init verdict is INTERPRETED.**"),
                }

    # ---- the directional call (doc 61 SS11, ruling 4) ---------------------
    head["directional_call_doc61_SS11"] = {
        "boolean": "amplitude_hurts_T1",
        "registered_expectation": True,
        "bar": "best amplitude T = 1 cell <= 0.9220",
        "measured": verdicts.get("amplitude_hurts_T1"),
        "held": (None if verdicts.get("amplitude_hurts_T1") is None
                 else bool(verdicts["amplitude_hurts_T1"] is True)),
        "rationale": (
            "(a) arXiv:2509.09541 evaluates ANGLE against AMPLITUDE encoding on a "
            "compositional-generalisation task with a classical DisCoCat comparator "
            "and reports ANGLE ENCODING GENERALLY STRONGER -- a directional precedent "
            "on this exact axis. (b) the parameter/sample arithmetic: 408 trainable "
            "parameters against 528 training items where the committed T = 1 register "
            "reaches 0.9820 with 264. (c) the programme's own width precedent runs "
            "the same way -- the wider register was the harder thing to train at "
            "every rung, and T = 1 is where the committed registers are strongest."),
        "the_alternative_is_fully_live_doc61_SS11_2": (
            "the boolean reads the BEST OF THE CELLS; A3 at 408 parameters is a MUCH "
            "MILDER overparameterisation than the brickwork's 1800 the call was "
            "priced against; the amplitude features carry strictly more information "
            "than F2's two coordinates; T = 1 is the regime in which EVERY committed "
            "encoding converted; and doc 58 SS19.1 has just demonstrated that an "
            "initialisation change alone moved a register arm by +0.157. **A SINGLE "
            "A3-IDENTITY CELL AT 0.93 REFUTES THE CALL.**"),
        "every_other_boolean": ("NO DIRECTION REGISTERED -- stated as IGNORANCE, not "
                                "as a hedge (doc 61 SS11.1)."),
        "programme_record": "1 held (twice-drawn) and 1 missed (doc 58 SS19.2)",
    }

    # ---- composition cost + the ladder (doc 61 SS10.5) -- DESCRIPTIVE -----
    cc = {}
    for rung in R.RUNGS:
        for ini in R.INITS:
            c1, c2 = "%s-T1-%s" % (rung, ini), "%s-T2-%s" % (rung, ini)
            if arms[c1].get("complete") and arms[c2].get("complete"):
                cc["%s-%s" % (rung, ini)] = arms[c1]["mean"] - arms[c2]["mean"]
            else:
                cc["%s-%s" % (rung, ini)] = None
    head["composition_cost_doc61_SS10_5"] = {
        "cc": cc,
        "definition": "cc(A, init) = heldout_mean(T=1) - heldout_mean(T=2)",
        "committed_ladder_doc56_SS18_1": COMMITTED_LADDER,
        "no_boolean": ("DESCRIPTIVE. doc 61 SS10.7(b) refuses any T1-T2 subtraction as "
                       "a criterion (doc 55 SS6.10b, inherited)."),
        "branch_B1": ("COST RISES WITH ENCODING COMPLEXITY -- the ladder's own "
                      "ordering is by encoding complexity and the amplitude register "
                      "is the most complex encoding the programme has built (7-15 "
                      "free coordinates, strongly entangled within itself). Predicts "
                      "cc(A3) >= 0.132 and cc(A4) >= cc(A3), plausibly both above "
                      "0.253."),
        "branch_B2": ("COST IS ABOUT THE MAP, NOT THE WIDTH -- dcc_map = -0.1207: the "
                      "eps-map HALVED the register's composition collapse at "
                      "bit-identical architecture (P-eps 0.132 against the old map's "
                      "0.253). The amplitude map has NO DEAD ZONES and NO RANGE "
                      "COMPRESSION, the two defects the eps-map existed to fix, so "
                      "its collapse could land at or below 0.132."),
        "no_direction_registered": (
            "doc 61 SS10.5: NO DIRECTION IS REGISTERED ON THE COMPOSITION COST, AND "
            "THIS IS STATED AS IGNORANCE RATHER THAN AS A HEDGE. B1 and B2 rest on "
            "two COMMITTED MEASUREMENTS THAT POINT IN OPPOSITE DIRECTIONS, and "
            "registering a direction would be picking one committed number over "
            "another with no mechanism to break the tie. Both branches were written "
            "in advance so whichever lands is a registered outcome and not a post-hoc "
            "narrative."),
    }

    # ---- bounds: CONTEXT ONLY, and they SATURATE (doc 61 SS12) ------------
    diag = json.load(open(R.AMP_DIAGNOSTIC, encoding="utf-8"))
    head["bounds_context_only_doc61_SS12"] = {
        "per_rung": {k: {"per_twin_bound": 1.0, "method": "EXACT -- saturation "
                                                          "certificate (rank 160 = "
                                                          "160 rows)",
                         "null": "1.0000 +- 0.0000, 200/200 draws saturate"}
                     for k in R.RUNGS},
        "source": os.path.basename(R.AMP_DIAGNOSTIC),
        "saturation_finding": diag.get("saturation_finding"),
        "no_criterion_reads_them": (
            "doc 61 SS12: NO CRITERION IN SS10 READS AGAINST THESE NUMBERS, and here "
            "they are not merely uninformative -- THEY ARE PROVABLY UNINFORMATIVE. At "
            "A3 and A4 the grade-1 reachable-accuracy bound is EXACTLY 1.0000 BY "
            "PROOF and so is every one of 200 size-matched null draws. THE INSTRUMENT "
            "THAT HAS GRADED EVERY RUNG OF THIS PROGRAMME DISCRIMINATES NOTHING AT "
            "THIS RUNG. **WHAT SATURATION DOES NOT SAY**: it does NOT say the "
            "encoding is expressive in any useful sense -- it is a statement about a "
            "1296- or 18496-dimensional linear model fitted to 160 constraints, the "
            "SAME degeneracy that makes an unregularised function-class-matched "
            "comparator vacuous (SS8.2). Both are consequences of ONE ARITHMETIC "
            "FACT."),
    }
    verdicts["bug_hunt_triggered"] = False
    reasons["bug_hunt_triggered"] = (
        "doc 61 SS12.1: STRUCTURALLY INERT AND REGISTERED AS SUCH. The standard gate "
        "fires when an arm exceeds its bound by > 0.06; both bounds are 1.0000, so "
        "bound + 0.06 > 1 and NO ACHIEVABLE ACCURACY CAN TRIGGER IT. Computed and "
        "stored for completeness; it is NOT EVIDENCE OF ANYTHING and no reading "
        "depends on it.")

    # ---- the comparator (doc 61 SS8) -------------------------------------
    if os.path.exists(comp_path):
        head["comparator"] = json.load(open(comp_path, encoding="utf-8"))
        head["comparator_status"] = "present"
    else:
        head["comparator"] = None
        head["comparator_status"] = (
            "MISSING -- run `python exp73_comparator.py` BEFORE this aggregator. doc "
            "61 SS8: tier (iii), the CAPACITY-MATCHED L2-regularised quadratic model, "
            "is the comparator every exp73 criterion that mentions a comparator "
            "reads, and SS8.3's restriction qualifier travels with it. Without it the "
            "arbiter rows of SS8.1 cannot be filled and the SS19 failure-mode "
            "checklist's 'if a cell fails while its tier (iii) comparator converts' "
            "branch cannot be evaluated.")

    # ---- ruling 2: THE A4 STAGE-2 TRIGGER, computed and printed ----------
    binds_t1 = [verdicts.get("binds_A3-T1-%s" % i) for i in R.INITS]
    init_eff = []
    for t in (1, 2):
        r = reasons.get("init_A3_T%d" % t)
        if isinstance(r, dict):
            init_eff.append(abs(r["init_delta"]))
    trig_binds = any(b is True for b in binds_t1)
    trig_init = any(x >= A4_INIT_TRIGGER for x in init_eff)
    decidable = (readable
                 and all(v is not None for v in binds_t1)
                 and len(init_eff) == 2)
    head["A4_stage2_trigger_ruling2"] = {
        "rule": ("orchestrator ruling 2: A4 RUNS ONLY IF A3 BINDS AT T = 1 OR SHOWS "
                 "AN INIT EFFECT >= 0.06 -- either makes the ladder point "
                 "informative. Stage 1 is A3-only."),
        "A3_binds_at_T1": {"per_init": {i: verdicts.get("binds_A3-T1-%s" % i)
                                        for i in R.INITS},
                           "any": trig_binds},
        "A3_init_effect_abs_delta": {"per_T": {("T%d" % t): (
            reasons["init_A3_T%d" % t]["init_delta"]
            if isinstance(reasons.get("init_A3_T%d" % t), dict) else None)
            for t in (1, 2)}, "max_abs": (max(init_eff) if init_eff else None),
            "threshold": A4_INIT_TRIGGER, "fires": trig_init},
        "decidable": bool(decidable),
        "A4_triggered": (bool(trig_binds or trig_init) if decidable else None),
        "if_triggered": (
            "A4 is 4 cells x 2 T x 5 seeds = 20 trained runs plus 20 matched floors "
            "at 552 parameters on 8 wires. Launch with "
            "`python exp73_run.py --arms A4-T1-uniform,A4-T1-identity,A4-T2-uniform,"
            "A4-T2-identity,S-shuffle-A4-T1-uniform,S-shuffle-A4-T1-identity,"
            "S-shuffle-A4-T2-uniform,S-shuffle-A4-T2-identity --worker i "
            "--nworkers 5`, then re-run the comparator with `--rungs A3,A4` and this "
            "aggregator. **doc 61 SS10.7(c): THERE IS NO A3-vs-A4 BOOLEAN** -- with 5 "
            "seeds and an 0.06 bar a two-way ranking between arms the grade-1 "
            "instrument cannot distinguish is a knife-edge magnet. The per-cell "
            "booleans are the verdicts and the ordering is DESCRIPTIVE."),
    }

    head["verdicts"] = verdicts
    head["verdict_reasons"] = reasons
    head["run_readable"] = bool(readable)
    head["gating_order_doc61_SS17_3_item11"] = (
        "anchor -> floors -> `binds` per cell -> `amplitude_helps/hurts` -> "
        "`init_helps/hurts` -> `cc` and SS10.5's ladder placement -> SS9's "
        "descriptives. NO CONTRAST IS READ PAST A MISSING OR FAILED FLOOR. A "
        "triggered cell whose floor has not been run has NO VERDICT and its booleans "
        "are stored as `null`, NEVER as `false` (SS15.4).")
    head["scope_doc61_SS13"] = [
        "NO QUANTUM READING OF ANY KIND. The readout IS a quadratic form and the "
        "matched classical model IS a quadratic-kernel model on the same features, "
        "in one line, with the equivalence already in the literature.",
        "NOT A CLAIM ABOUT THE QUADRATIC CLASS. A circuit that beats the "
        "CAPACITY-MATCHED tier (iii) comparator has beaten a REGULARISED classical "
        "model, not the function class, and that qualifier travels in the same "
        "sentence.",
        "NOT A STATEMENT ABOUT AMPLITUDE ENCODING IN GENERAL. TRUNCATED amplitude "
        "encoding at d = 8 (and d = 16 if stage 2 runs), two referents, two events, "
        "one category, one surface, 24 nouns, IN SIMULATION. The full 300-dim map is "
        "deferred to D3 and the state preparation's own cost is not measured.",
        "NOT A STATEMENT ABOUT BARREN PLATEAUS. 6- and 8-wire circuits, no "
        "gradient-variance-versus-qubit-count scaling measured.",
        "NOT A STATEMENT ABOUT THE ARCHITECTURE BEYOND THE MONOLITHIC BLOCK. The "
        "brickwork is priced and NOT RUN; a null on Circuit4(2q,3) is not a null on "
        "the brickwork.",
        "NOT 'MEANING', not composition-in-general, not DisCoCirc's expressive "
        "limits. The bounds are 1.0000 and always were.",
        "NOT NOVEL-FILLER GENERALISATION. exp69's held-out-PAIRING split, inherited.",
        "NOT A STATEMENT ABOUT ARAVEC. PC1..PC8 of a 300-dim vector over 44 words.",
        "ONE DATASET, ONE TASK FAMILY, TWO REFERENTS, TWO EVENTS, 5 SEEDS. doc 50 "
        "SS7's ceiling stands.",
        "THE CAPACITY HALF OF THE USER'S QUESTION WAS ANSWERED NEGATIVELY BY THE "
        "PRE-WORK, BEFORE THE COMPUTE WAS SPENT (doc 61 SS1.4 / SS20.1). Whatever "
        "exp73 measures is A FACT ABOUT TRAINING, NOT ABOUT REPRESENTATIONAL "
        "CAPACITY. That is a smaller experiment than 'put the whole vector on the "
        "wire' sounds like, and the doc says so rather than letting the framing "
        "survive the finding.",
        "SAMPLE COMPLEXITY CANNOT BE EXCLUDED (doc 61 SS19 row 3, SS20.2). At 1296 "
        "and 18496 joint dimensions on 660 items the function-class-matched "
        "comparator reaches 1.0000 BY DIMENSION COUNT, which is exactly what a "
        "sample-starved regime looks like from the inside. Tier (iii) is a "
        "MITIGATION, NOT A RESOLUTION.",
    ]
    head["confirm_draw_obligation_doc61_SS17_2"] = (
        "ON ANY POSITIVE HEADLINE: doc 47 SS18 is standing -- a confirm draw at "
        "generator seed + 1000, re-running the COMMITTED `exp69_datagen.py` "
        "UNMODIFIED, with ALL ARMS OF THE AFFECTED WIDTH PRESENT (both T, both "
        "inits), the full V-suite and every SS7 assertion re-run, and the S-shuffle "
        "floors RE-DRAWN matched to the fresh draw. No new surface template, no new "
        "noun, no new verb; the native-speaker gate stays closed. There is no "
        "warm-start donor here, so doc 58 SS16.2's donor-retraining clause does not "
        "apply.")
    head["wall_time_sec"] = time.time() - t0

    json.dump(head, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # Rule 3: the floors also get their OWN registered file (doc 61 SS16.5), because
    # stage 2 may run later than stage 1 and nothing may be overwritten.
    json.dump({"component": "exp73 procedure-matched floors (doc 61 SS15)",
               "spec": SPEC, "aggregated_from": a.ckpt_dir,
               "dataset_sha256": store[1]["dataset_sha256"],
               "floors": floors, "floor_clearance": clear,
               "construction": ("the committed S-shuffle "
                                "(`exp69_run.relabel_shuffle_q`, imported unchanged), "
                                "bearer designation reassigned PER PAIR independently "
                                "of the features, same items, same splits, same 5 "
                                "seeds, run under the IDENTICAL ARM PROCEDURE. THE "
                                "INIT IS PART OF THE PROCEDURE, so each cell carries "
                                "its OWN floor (doc 61 SS15.1)."),
               "rule3_note": ("separate file because stage 2 may run later than stage "
                              "1 and Rule 3 forbids overwriting (doc 61 SS16.5)"),
               "wall_time_sec": time.time() - t0},
              open(a.floors_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ print
    print("\n=== exp73 -- doc 61, THE AMPLITUDE RUNG ===")
    print("anchor_replicates_committed = %s   (run_readable = %s)"
          % (verdicts.get("anchor_replicates_committed"), head["run_readable"]))
    print("\n%-24s %8s %8s   %8s %8s   %s"
          % ("cell", "mean", "sd", "floor", "floorsd", "binds"))
    for c in cells:
        ab, fb = arms[c], floors[c]
        if not ab.get("complete"):
            print("%-24s   INCOMPLETE (%d/%d seeds)"
                  % (c, ab.get("n_runs", 0), a.seeds))
            continue
        print("%-24s %8.4f %8.4f   %8s %8s   %s"
              % (c, ab["mean"], ab["sd"],
                 ("%.4f" % fb["mean"]) if fb.get("complete") else "--",
                 ("%.4f" % fb["sd"]) if fb.get("complete") else "--",
                 verdicts.get("binds_%s" % c)))
    for t in (1, 2):
        r = reasons.get("amplitude_T%d" % t)
        if isinstance(r, dict):
            print("\nT=%d  best amplitude %s = %.4f vs best committed %.4f  "
                  "delta %+.4f  -> helps=%s hurts=%s"
                  % (t, r["best_amplitude_cell"], r["best_amplitude_mean"],
                     r["best_committed_2qubit_mean"], r["delta"],
                     verdicts.get("amplitude_helps_T%d" % t),
                     verdicts.get("amplitude_hurts_T%d" % t)))
    print("\ncomposition cost cc(T1 - T2): %s"
          % {k: (None if v is None else round(v, 4)) for k, v in cc.items()})
    print("A4 stage-2 trigger (ruling 2): decidable=%s triggered=%s"
          % (head["A4_stage2_trigger_ruling2"]["decidable"],
             head["A4_stage2_trigger_ruling2"]["A4_triggered"]))
    print("comparator: %s" % head["comparator_status"])
    print("\n[exp73-agg] wrote %s and %s (%.0fs)"
          % (a.out, a.floors_out, head["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
