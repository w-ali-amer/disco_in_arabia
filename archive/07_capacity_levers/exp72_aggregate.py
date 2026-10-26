# -*- coding: utf-8 -*-
"""exp72_aggregate.py -- doc 58's ONLY results writer (doc 53 SS7.3's rule: the
aggregator ships in the same commit as the runner).

IT COMPUTES NOTHING NEW. Every number here is read from the per-run checkpoint
JSONs `exp72_run.py` wrote, or imported from the committed `results_exp69.json` /
`results_exp70.json`. The bars are recomputed at full precision from the imported
committed values and cross-checked against doc 58 SS9.2's display table.

THE VERDICTS ARE READ IN THE REGISTERED ORDER (doc 58 SS16.3 step 10):

    anchors -> floors -> rescue_full / rescue_partial per cell
            -> positive_control_degrades -> SS8's descriptives

and **no contrast is read past a missing or failed floor**. A cell that reached
its `rescue_partial` bar but whose procedure-matched floor has not been run has
NO VERDICT: its booleans are stored as `null`, NEVER as FALSE (doc 58 SS14.5).

    python exp72_aggregate.py                 # stage 1 -> results_exp72.json
    python exp72_aggregate.py --stage2        # + results_exp72_floors_stage2.json
"""
import argparse
import json
import math
import os
import platform
import time

import numpy as np
import torch

import exp72_run as X                                   # noqa: E402

RESULTS_BASE = "results_exp72"
FLOORS_BASE = "results_exp72_floors_stage2"
FLOOR_SD = 2.0                                          # doc 58 SS9.2


# ------------------------------------------------------------------ plumbing
def load_runs():
    """Every per-run checkpoint exp72 has written, keyed by label."""
    out = {}
    if not os.path.isdir(X.CKPT_DIR):
        return out
    for fn in sorted(os.listdir(X.CKPT_DIR)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(X.CKPT_DIR, fn), encoding="utf-8"))
        out[r.get("label") or fn[:-5]] = r
    return out


def block(rows, extras=True):
    """The programme's per-arm block: mean, sample sd (ddof = 1), per seed."""
    rr = sorted(rows, key=lambda x: x["seed"])
    v = [r["heldout_test_acc"] for r in rr]
    b = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
         "per_seed": v, "seeds": [r["seed"] for r in rr], "n_seeds": len(rr)}
    if extras:
        b.update({
            "val_per_seed": [r["val_acc"] for r in rr],
            "train_per_seed": [r["train_acc"] for r in rr],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rr],
            "selected_stage_per_seed": [r.get("selected_stage") for r in rr],
            "val_minus_heldout_per_seed": [r["val_minus_heldout"] for r in rr],
            "mean_val_minus_heldout": float(np.mean(
                [r["val_minus_heldout"] for r in rr])),
            "max_epochs": rr[0]["max_epochs"], "n_params": rr[0]["n_params"],
            "wires": rr[0]["wires"], "n_test": rr[0]["n_test"],
            "wall_sec_total": float(sum(r["wall_sec"] for r in rr)),
            "ckpt_sha256": [r["ckpt_sha256"] for r in rr],
        })
    return b


def clearance(arm_blk, floor_blk):
    """doc 58 SS9.3, doc 55 SS6.6 inherited VERBATIM: the floor's own SD is
    PRIMARY, all three readings are computed and reported, and an arm whose
    readings DISAGREE is reported as disagreeing rather than resolved silently."""
    d = arm_blk["mean"] - floor_blk["mean"]
    pooled = float(math.sqrt((arm_blk["sd"] ** 2 + floor_blk["sd"] ** 2) / 2.0))
    readings = {
        "floor_sd": {"sd": floor_blk["sd"],
                     "bar": floor_blk["mean"] + FLOOR_SD * floor_blk["sd"],
                     "clears": bool(arm_blk["mean"]
                                    >= floor_blk["mean"]
                                    + FLOOR_SD * floor_blk["sd"])},
        "arm_sd": {"sd": arm_blk["sd"],
                   "bar": floor_blk["mean"] + FLOOR_SD * arm_blk["sd"],
                   "clears": bool(arm_blk["mean"]
                                  >= floor_blk["mean"]
                                  + FLOOR_SD * arm_blk["sd"])},
        "pooled_sd": {"sd": pooled,
                      "bar": floor_blk["mean"] + FLOOR_SD * pooled,
                      "clears": bool(arm_blk["mean"]
                                     >= floor_blk["mean"] + FLOOR_SD * pooled)},
    }
    return {"arm_mean": arm_blk["mean"], "floor_mean": floor_blk["mean"],
            "delta_over_floor": d,
            "sd_units_over_floor_using_floor_sd":
                (d / floor_blk["sd"]) if floor_blk["sd"] > 0 else float("inf"),
            "readings": readings, "primary_reading": "floor_sd",
            "clears_floor": readings["floor_sd"]["clears"],
            "clears_under_all_readings":
                bool(all(v["clears"] for v in readings.values())),
            "clears_under_any_reading":
                bool(any(v["clears"] for v in readings.values())),
            "readings_agree":
                bool(len({v["clears"] for v in readings.values()}) == 1)}


def strat(rows, keyfn):
    cell = {}
    for r in rows:
        k = keyfn(r)
        cell.setdefault(k, [0, 0])
        cell[k][0] += int(r["correct"])
        cell[k][1] += 1
    return {str(k): {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(cell.items())}


def purity_profile(runs):
    """doc 58 SS8, and doc 55 SS18.3 / doc 56 SS15.1-SS15.2's conventions VERBATIM.

    THE REGISTERED QUESTION IS DESCRIPTIVE AND CARRIES NO BOOLEAN AND NO CRITERION
    (SS8.2 / SS8.3): *does a successful rescue restore the q-asymmetry undo
    signature to the healthy arms' profile?* The healthy profile is committed
    (doc 55 SS20.5): on correct q = 2 items the readout restores per-wire purity
    to its post-event-1 level (R2-A 0.8755 vs 0.8751); on q = 1 items it does not
    (0.8017 vs 0.8953). doc 56 SS15.2's amendment applies to I2: for the ENTANGLED
    arms the restoration target is the INTRO purity, not 1. Stratified by q and by
    correctness, exactly as doc 55 SS19.4 registered it."""
    rep, acc = {}, {}
    for r in runs:
        for row in r["heldout_items"]:
            rep.setdefault(row["id"], row)
            acc.setdefault(row["id"], []).append(int(row["correct"]))
    cells = {}
    for iid, row in rep.items():
        q = row["query_index"]
        ok = bool(np.mean(acc[iid]) >= 0.5)
        k = "q%d_%s" % (q, "correct" if ok else "incorrect")
        c = cells.setdefault(k, {"n": 0, "stages": {}})
        c["n"] += 1
        for stage, vals in row["purity"].items():
            c["stages"].setdefault(stage, []).append(float(np.mean(vals)))
    cell_out = {k: {"n": c["n"],
                    "mean_wire_purity": {s: float(np.mean(v))
                                         for s, v in sorted(c["stages"].items())}}
                for k, c in sorted(cells.items())}
    restore = {}
    for q in (1, 2):
        for corr in (True, False):
            ids = [i for i in rep
                   if rep[i]["query_index"] == q
                   and (np.mean(acc[i]) >= 0.5) == corr
                   and "after_event1" in rep[i]["purity"]]
            if not ids:
                continue
            g1 = [abs(float(np.mean(rep[i]["purity"]["at_readout"]))
                      - float(np.mean(rep[i]["purity"]["after_event1"])))
                  for i in ids]
            gi = [abs(float(np.mean(rep[i]["purity"]["at_readout"]))
                      - float(np.mean(rep[i]["purity"]["after_intro"])))
                  for i in ids]
            restore["q%d_%s" % (q, "correct" if corr else "incorrect")] = {
                "n": len(ids),
                "readout_vs_after_event1_mean_abs_gap": float(np.mean(g1)),
                "readout_vs_INTRO_mean_abs_gap": float(np.mean(gi)),
                "mean_readout_purity": float(np.mean(
                    [float(np.mean(rep[i]["purity"]["at_readout"])) for i in ids])),
                "mean_after_event1_purity": float(np.mean(
                    [float(np.mean(rep[i]["purity"]["after_event1"]))
                     for i in ids])),
                "mean_intro_purity": float(np.mean(
                    [float(np.mean(rep[i]["purity"]["after_intro"]))
                     for i in ids])),
            }
    return {"cells": cell_out, "restoration": restore,
            "q_asymmetry_undo_gap_correct_items": (
                (restore.get("q1_correct", {}).get(
                    "readout_vs_after_event1_mean_abs_gap")),
                (restore.get("q2_correct", {}).get(
                    "readout_vs_after_event1_mean_abs_gap"))),
            "status": ("DESCRIPTIVE -- doc 58 SS8.2 / SS8.3. No boolean, no "
                       "criterion, and it MAY NOT be promoted onto a criterion "
                       "after results exist (the exp44 no-promotion clause).")}


def curve_summary(runs):
    """doc 58 SS9.7's 'per-arm training curves', summarised rather than dumped."""
    out = []
    for r in sorted(runs, key=lambda x: x["seed"]):
        c = r.get("curve") or []
        if not c:
            out.append({"seed": r["seed"]})
            continue
        v = [p["val_acc"] for p in c]
        marks = sorted({1, len(c) // 4, len(c) // 2, (3 * len(c)) // 4, len(c)})
        out.append({
            "seed": r["seed"], "n_epochs": len(c),
            "val_at_epochs": {str(e): c[e - 1]["val_acc"] for e in marks},
            "lr_at_epochs": {str(e): c[e - 1]["lr"] for e in marks},
            "best_val": float(max(v)), "final_val": v[-1],
            "selected_epoch": r["selected_epoch"],
            "selected_stage": r.get("selected_stage"),
            "selected_fraction_of_budget": r["selected_epoch"] / float(len(c)),
            "epochs_after_selection": len(c) - r["selected_epoch"],
        })
    return out


# ================================================================ the verdicts
def cell_verdict(inst, rescue, arm_blk, bar, floor_blk, floor_source):
    """doc 58 SS9.2's two booleans, and SS14.5's GATING ORDER.

      stage-1 mean -> trigger test (SS14.2) -> stage-2 floor if triggered ->
      ONLY THEN rescue_full / rescue_partial.

    A TRIGGERED CELL WHOSE FLOOR HAS NOT BEEN RUN HAS NO VERDICT and its booleans
    are `null`, never FALSE."""
    mean = arm_blk["mean"]
    triggered = bool(mean >= bar["rescue_partial_bar"])
    out = {"instance": inst, "rescue": rescue,
           "heldout_mean": mean, "heldout_sd": arm_blk["sd"],
           "per_seed": arm_blk["per_seed"],
           "baseline": bar["baseline"], "comparator": bar["comparator"],
           "gap": bar["gap"],
           "rescue_full_bar": bar["rescue_full_bar"],
           "rescue_partial_bar": bar["rescue_partial_bar"],
           "reaches_full_bar": bool(mean >= bar["rescue_full_bar"]),
           "reaches_partial_bar": bool(mean >= bar["rescue_partial_bar"]),
           "per_seed_rescue_delta": [v - bar["baseline"]
                                     for v in arm_blk["per_seed"]],
           "mean_rescue_delta": mean - bar["baseline"],
           "stage2_triggered": triggered}
    if not triggered:
        cl = clearance(arm_blk, floor_blk)
        out.update({
            "floor_used": floor_source, "floor_status": "COMMITTED (orientation)",
            "floor_clearance_orientation_only": cl,
            "rescue_full": False, "rescue_partial": False,
            "verdict": ("NOT RESCUED. doc 58 SS14.2: a cell that does not reach its "
                        "rescue_partial bar cannot pass either boolean regardless "
                        "of its floor, so no procedure-matched floor is run and the "
                        "stage-1 comparison against the COMMITTED floor is stated "
                        "as ORIENTATION ONLY."),
        })
        return out
    if floor_blk is None or floor_source is None or floor_blk.get(
            "procedure_matched") is not True:
        out.update({
            "floor_used": None, "floor_status": "MISSING -- NOT YET RUN",
            "rescue_full": None, "rescue_partial": None,
            "verdict": ("TRIGGERED, NO VERDICT. doc 58 SS14.5: this cell reached its "
                        "rescue_partial bar, so a floor RE-RUN UNDER ITS OWN RESCUE "
                        "PROCEDURE is required before either boolean may be read. "
                        "The booleans are stored as null, NEVER as FALSE. Run "
                        "`python exp72_run.py --stage2 --cells %s:%s`."
                        % (inst, rescue)),
        })
        return out
    cl = clearance(arm_blk, floor_blk)
    clears = cl["clears_floor"]
    contaminated = bool(not clears)
    out.update({
        "floor_used": floor_source, "floor_status": "PROCEDURE-MATCHED (stage 2)",
        "floor_mean": floor_blk["mean"], "floor_sd": floor_blk["sd"],
        "floor_per_seed": floor_blk["per_seed"],
        "floor_clearance": cl,
        "rescue_full": bool(mean >= bar["rescue_full_bar"] and clears),
        "rescue_partial": bool(mean >= bar["rescue_partial_bar"] and clears),
        "floor_contaminated": contaminated,
        "verdict": ("FLOOR-CONTAMINATED -- doc 58 SS10: the procedure is fitting the "
                    "SPLIT, not the labels; the rescue is NOT READ and the cell is "
                    "reported with both numbers." if contaminated else
                    "READ against a procedure-matched floor (doc 58 SS14.3)."),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2", action="store_true",
                    help="also write the stage-2 floors file")
    a = ap.parse_args()
    t0 = time.time()

    runs = load_runs()
    anc_committed = X.committed_anchor_values()
    tgt = X.comparator_targets()
    bar = X.bars()

    # ---- the anchor file (doc 58 SS5, BLOCKING, read first)
    anchor_files = sorted((p for p in os.listdir(".")
                           if p.startswith(X.ANCHOR_BASE) and p.endswith(".json")),
                          key=os.path.getmtime)
    anchors = json.load(open(anchor_files[-1], encoding="utf-8")) \
        if anchor_files else None
    anchors_ok = bool(anchors and anchors.get("anchors_replicate_committed"))

    # ---- stage-1 arms
    arms, floors_stage2, missing = {}, {}, []
    for inst in ("I1", "I2"):
        for rescue in X.RESCUES:
            rr = [r for r in runs.values()
                  if r.get("kind") == "rescue" and r.get("instance") == inst
                  and r.get("cell_rescue") == rescue]
            if len(rr) != X.N_SEEDS:
                missing.append("%s:%s (%d/%d runs)" % (inst, rescue, len(rr),
                                                       X.N_SEEDS))
                continue
            arms["%s|%s" % (inst, rescue)] = block(rr)
            fr = [r for r in runs.values()
                  if r.get("kind") == "floor" and r.get("instance") == inst
                  and r.get("cell_rescue") == rescue]
            if len(fr) == X.N_SEEDS:
                fb = block(fr)
                fb["procedure_matched"] = True
                fb["rescue_procedure"] = rescue
                floors_stage2["%s|%s" % (inst, rescue)] = fb
    pc = [r for r in runs.values() if r.get("kind") == "control"]
    pc_blk = block(pc) if len(pc) == X.N_SEEDS else None
    if pc_blk is None:
        missing.append("PC:%s (%d/%d runs)" % (X.PC_RESCUE, len(pc), X.N_SEEDS))

    # ---- the 16 booleans, in the registered gating order
    verdicts, booleans = {}, {}
    for inst in ("I1", "I2"):
        committed_floor = {"mean": anc_committed[inst]["floor_mean"],
                           "sd": anc_committed[inst]["floor_sd"],
                           "per_seed": anc_committed[inst]["floor_per_seed"],
                           "procedure_matched": False}
        for rescue in X.RESCUES:
            key = "%s|%s" % (inst, rescue)
            if key not in arms:
                verdicts[key] = {"instance": inst, "rescue": rescue,
                                 "rescue_full": None, "rescue_partial": None,
                                 "verdict": "NOT RUN"}
                booleans["rescue_full(%s, %s)" % (inst, rescue)] = None
                booleans["rescue_partial(%s, %s)" % (inst, rescue)] = None
                continue
            f2 = floors_stage2.get(key)
            v = cell_verdict(inst, rescue, arms[key], bar[inst],
                             f2 if f2 else committed_floor,
                             ("stage-2 procedure-matched floor" if f2 else
                              "committed matched floor (doc 58 SS14.1, "
                              "ORIENTATION ONLY)"))
            verdicts[key] = v
            booleans["rescue_full(%s, %s)" % (inst, rescue)] = v["rescue_full"]
            booleans["rescue_partial(%s, %s)" % (inst, rescue)] = v["rescue_partial"]

    triggered = sorted(k for k, v in verdicts.items()
                       if v.get("stage2_triggered"))
    unfloored = sorted(k for k, v in verdicts.items()
                       if v.get("stage2_triggered")
                       and v.get("rescue_full") is None)

    # ---- doc 58 SS9.4, the positive control
    pcv = None
    if pc_blk:
        deg = X.PC_COMMITTED_MEAN - pc_blk["mean"]
        fires_doc = bool(pc_blk["mean"] < X.PC_DEGRADE_BAR)
        exact_bar = X.PC_COMMITTED_MEAN - 0.06
        fires_exact = bool(pc_blk["mean"] < exact_bar)
        pcv = {"arm": X.PC_ARM, "committed_arm": X.PC_COMMITTED_ARM,
               "rescue": X.PC_RESCUE,
               "committed_mean": X.PC_COMMITTED_MEAN,
               "committed_mean_full_precision": anc_committed["PC"]["mean"],
               "exp72_mean": pc_blk["mean"], "exp72_sd": pc_blk["sd"],
               "per_seed": pc_blk["per_seed"],
               "degradation": deg,
               "bar_doc58_literal": X.PC_DEGRADE_BAR,
               "bar_full_precision": exact_bar,
               "positive_control_degrades": fires_doc,
               "positive_control_degrades_full_precision_bar": fires_exact,
               "knife_edge": bool(fires_doc != fires_exact),
               "selected_epoch_per_seed": pc_blk["selected_epoch_per_seed"],
               "committed_selected_epoch_per_seed": [121, 121, 14, 99, 114],
               "reading": ("doc 58 SS9.4 / SS15.1: a rescue procedure that DEGRADES "
                           "a healthy arm by more than 0.06 is FLAGGED, and every "
                           "rescue read under that procedure is read with suspicion "
                           "and reported WITH THE FLAG ATTACHED IN THE SAME "
                           "SENTENCE. It is NOT a kill switch -- it is a warning "
                           "that travels (exp43c precedent, doc 24 SS5.2: a control "
                           "that cannot fail is not a control)."),
               "no_floor_doc58_SS15_2": (
                   "the control carries NO floor and that is a decision, not an "
                   "omission: it is a DEGRADATION CHECK against a committed value, "
                   "not a gain claim, and doc 52 SS3's 'a gain criterion needs a "
                   "null' does not bind a loss check.")}
        booleans["positive_control_degrades"] = fires_doc

    # ---- doc 58 SS9.5, the inert bug-hunt gate
    bug = {"bug_hunt_triggered": False,
           "status": "STRUCTURALLY INERT, and registered as such",
           "reason": ("doc 58 SS9.5, orchestrator ruling 5 (KEPT and stored as "
                      "inert -- a documented cannot-fire beats an absent row). The "
                      "gate fires when an arm exceeds its context bound by > 0.06. "
                      "BOTH instances' per-twin bounds are 1.0000 (doc 55 SS12 for "
                      "F2, doc 56 SS10 for N-in), so bound + 0.06 > 1 and NO "
                      "achievable accuracy can trigger it. The positive control's "
                      "FA bound of record is 0.9625, so its gate would fire above "
                      "1.0225 -- also unreachable. Computed and stored for "
                      "completeness; it is not evidence of anything and no reading "
                      "depends on it."),
           "bounds_context_only_doc58_SS11": {
               "I1": {"bound": 1.0, "fraction": "160/160",
                      "source": "diagnostic_sb_bounds.json, doc 55 SS12"},
               "I2": {"bound": 1.0, "fraction": "160/160",
                      "source": "diagnostic_entangled_bounds.json, doc 56 SS10"},
               "PC": {"bound": 0.9625, "fraction": "154/160",
                      "source": "diagnostic_readout_arity_bounds_addendum.json"},
               "note": ("NO CRITERION READS THESE. Both instance bounds are 1.0000 "
                        "and IDENTICAL, so grade 1 discriminates nothing between "
                        "them and cannot bound any rescue from above in a useful "
                        "way. The experiment rests entirely on held-out measurement "
                        "against the committed comparator.")}}
    booleans["bug_hunt_triggered"] = False

    # ---- doc 58 SS9.6, THE ONE DIRECTIONAL CALL
    dc_key = "I1|R-warmstart"
    dcv = verdicts.get(dc_key, {})
    call = {
        "cell": "rescue_partial(I1, R-warmstart)",
        "registered_expectation": True,
        "registered_before_the_run": True,
        "outcome": dcv.get("rescue_partial"),
        "heldout_mean": dcv.get("heldout_mean"),
        "bar": bar["I1"]["rescue_partial_bar"],
        "lands": (None if dcv.get("rescue_partial") is None
                  else bool(dcv.get("rescue_partial") is True)),
        "rationale_doc58_SS9_6": (
            "the same-family T = 1 solution EXISTS, at 0.9820, on the identical "
            "features, identical architecture and identical 264-parameter set. If "
            "composition is a FINE-TUNING problem rather than a representation "
            "problem, the T = 1 optimum is the initialisation most likely to sit in "
            "the T = 2 basin's neighbourhood. External support: arXiv:2409.08777 "
            "reports avoiding barren plateaus in DisCoCirc by pretraining "
            "components. The call is CONFINED to the warm-start cell because the "
            "in-programme precedents cut the other way for budget-type levers (doc "
            "38 / exp54: 2.5x budget did not rescue) and because doc 58 SS4.3's own "
            "literature registers the failure mode -- transfer into a dissimilar "
            "basin -- that SS6.4.3 shows this transfer is exposed to."),
        "if_it_lands_doc58_SS10": (
            "composition is a FINE-TUNING problem for the product register: the "
            "T = 1 solution is in the T = 2 basin's neighbourhood and standard "
            "training simply does not find it from a cold start. THIS WOULD BE THE "
            "RUN'S HEADLINE -- and it MAY NOT BE CLAIMED ON THE FIRST DRAW ALONE. "
            "The confirm draw at generator seed + 1000 with ALL ARMS OF I1 PRESENT "
            "(its anchor and all four rescues), plus doc 58 SS16.2's RETRAINED "
            "DONOR, is MANDATORY (doc 47 SS18, doc 58 SS16.3 step 11). Not a "
            "statement about DisCoCirc's expressive limits, and not about I2."),
        "every_other_cell": ("NO DIRECTION REGISTERED -- stated as ignorance, not as "
                             "a hedge (doc 58 SS9.6). 15 booleans."),
    }

    # ---- doc 58 SS9.7, reported not registered
    reported = {}
    for key, blk in arms.items():
        inst, rescue = key.split("|")
        rr = [runs[k] for k in runs
              if runs[k].get("kind") == "rescue"
              and runs[k].get("instance") == inst
              and runs[k].get("cell_rescue") == rescue]
        rows = [row for r in rr for row in r["heldout_items"]]
        reported[key] = {
            "per_q_accuracy": strat(rows, lambda r: "q%d" % r["query_index"]),
            "per_surface_order_accuracy": strat(
                rows, lambda r: "|".join(r["order_flags"])),
            "per_queried_verb_accuracy": strat(rows,
                                               lambda r: r["queried_verb"]),
            "val_minus_heldout_per_seed": blk["val_minus_heldout_per_seed"],
            "selected_epoch_per_seed": blk["selected_epoch_per_seed"],
            "selected_fraction_of_budget": [
                e / float(blk["max_epochs"])
                for e in blk["selected_epoch_per_seed"]],
            "per_seed_rescue_delta": verdicts[key]["per_seed_rescue_delta"],
            "training_curves": curve_summary(rr),
            "purity_doc58_SS8": purity_profile(rr),
        }

    # anchors' own purity profile, so "restored to the healthy profile?" has its
    # unrescued reference on the SAME instance (doc 58 SS8.1: recorded on all
    # rescue arms AND both anchors).
    for inst in ("I1", "I2"):
        ar = [r for r in runs.values()
              if (r.get("label") or "").startswith("%s_anchor_s" % inst)]
        if len(ar) == X.N_SEEDS:
            reported["%s|anchor" % inst] = {
                "purity_doc58_SS8": purity_profile(ar),
                "training_curves": curve_summary(ar),
                "per_q_accuracy": strat(
                    [row for r in ar for row in r["heldout_items"]],
                    lambda r: "q%d" % r["query_index"])}

    # ---- doc 58 SS7.2's cross-run failing direction for R-warmstart
    ws = {}
    for inst in ("I1", "I2"):
        cks = [os.path.join(X.CKPT_DIR, "%s_R-warmstart_s%d.pt" % (inst, s))
               for s in range(X.N_SEEDS)]
        if all(os.path.exists(p) for p in cks):
            states = {s: torch.load(cks[s], map_location="cpu")
                      for s in range(X.N_SEEDS)}
            ws[inst] = X.check_warmstart_spread(states)
            ws[inst]["note"] = (
                "compared on the SELECTED checkpoints (the object every held-out "
                "number is read from). doc 58 SS6.0 / SS7.2's failing direction: "
                "`seed` reshuffles the training order every epoch, so five "
                "R-warmstart runs of a cell must NOT be bit-identical.")

    out = {
        "experiment": ("exp72 -- doc 58, THE RESCUE RUNG: do the field's proposed "
                       "trainability mitigations rescue instances whose "
                       "information content is independently certified "
                       "extractable?"),
        "spec": X.SPEC, "spec_commit_line": X.spec_commit(),
        "aggregator": "exp72_aggregate.py -- the ONLY results writer (doc 53 SS7.3)",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "sd_convention": "sample sd, ddof = 1",
        "dataset": {"path": X.R70.DATA_A,
                    "sha256": X.DATASET_SHA_DOC58,
                    "no_datagen": ("doc 58 SS16.1: exp72 generates NO Arabic and "
                                   "runs no V-suite; the dataset is exp69's Part A, "
                                   "byte-identical, sha-pinned and already "
                                   "V-suite-cleared, so doc 50 SS5.2's "
                                   "native-speaker gate does not reopen.")},
        "instances": {i: {k: v for k, v in X.INSTANCES[i].items()
                          if k != "comparator_tier"} for i in X.INSTANCES},
        "committed_values": anc_committed,
        "comparator_targets_doc58_SS3": tgt,
        "bars_doc58_SS9_2": bar,

        # ---------------------------------------------------- 1. THE ANCHORS
        "anchors_doc58_SS5": (anchors or {"status": "NOT RUN"}),
        "anchors_replicate_committed": anchors_ok,
        "gating_order_doc58_SS16_3_step10": (
            "anchors -> floors -> rescue_full / rescue_partial per cell -> "
            "positive_control_degrades -> SS8's descriptives. NO CONTRAST IS READ "
            "PAST A MISSING OR FAILED FLOOR."),

        # ---------------------------------------------------- 2. THE ARMS
        "arms": arms,
        "positive_control": pcv,
        "runs_missing": missing,

        # ---------------------------------------------------- 3. THE FLOORS
        "floors_doc58_SS14": {
            "stage1_committed_floors": {
                i: {"mean": anc_committed[i]["floor_mean"],
                    "sd": anc_committed[i]["floor_sd"],
                    "per_seed": anc_committed[i]["floor_per_seed"],
                    "source": anc_committed[i]["source"]} for i in ("I1", "I2")},
            "stage1_is_orientation_only": (
                "doc 58 SS14.1: all rescue arms are compared FIRST to the committed "
                "matched floors. THIS COMPARISON CARRIES NO VERDICT. Its sole "
                "function is to decide which cells go to stage 2."),
            "stage2_trigger_doc58_SS14_2": (
                "any cell whose held-out mean reaches its rescue_partial bar "
                "(I1 >= 0.864667, I2 >= 0.823333) TRIGGERS a matched floor RE-RUN "
                "UNDER THAT RESCUE PROCEDURE. A cell that does not reach its "
                "partial bar cannot pass either boolean regardless of its floor."),
            "stage2_floors": floors_stage2,
            "triggered_cells": triggered,
            "triggered_but_unfloored": unfloored,
            "severed_donor_protocol_doc58_SS14_4": {
                "rule": ("a warm start injects information through the "
                         "INITIALISATION, so THE DONOR'S LABELS ARE SEVERED TOO: "
                         "the T = 1 donor arm is retrained on the shuffled bearer "
                         "designation at the committed T = 1 recipe (150 epochs), "
                         "its checkpoint is selected on the SHUFFLED val accuracy, "
                         "and the shuffled T = 2 run is warm-started from it."),
                "seed_linkage": ("REGISTERED: floor seed k's shuffle RNG seed "
                                 "drives BOTH the T = 1 donor's per-pair "
                                 "reassignment AND the T = 2 run's -- one seed, two "
                                 "item sets."),
                "BUILD_LEVEL_OPERATIONALISATION_FLAGGED_FOR_THE_ORCHESTRATOR": (
                    "doc 58 SS14.4 registers BOTH a five-seed donor retraining AND "
                    "seed linkage, and the two only close under one reading: the "
                    "T = 2 floor run at seed k is warm-started from THE DONOR "
                    "TRAINED AT SEED k UNDER SHUFFLE SEED k. That reading is the "
                    "only one consistent with SS15.3's count of 5 donor runs per "
                    "cell AND with the committed per-seed S-shuffle construction; "
                    "under it, step 2's 'select the donor on the SHUFFLED val "
                    "accuracy' is the WITHIN-RUN checkpoint selection (the content "
                    "that matters: no true label reaches the initialisation) and "
                    "the cross-seed lowest-index tie-break is inert. A cross-seed "
                    "donor selection would either break the linkage or cost 25 "
                    "donor runs per cell, neither of which is registered. THIS IS A "
                    "BUILD-LEVEL RESOLUTION OF A REGISTERED AMBIGUITY AND IS "
                    "REPORTED RATHER THAN BURIED."),
            },
            "gating_order_doc58_SS14_5": (
                "stage-1 mean -> trigger test -> stage-2 floor if triggered -> ONLY "
                "THEN the booleans. A TRIGGERED CELL WHOSE FLOOR HAS NOT BEEN RUN "
                "HAS NO VERDICT and its booleans are stored as null, NEVER FALSE."),
            "sd_readings_doc58_SS9_3": (
                "'clears its procedure-matched floor by >= 2 SD' uses the FLOOR'S "
                "OWN SD AS PRIMARY, with all three readings (floor SD, arm SD, "
                "pooled) computed and reported, and an arm whose readings DISAGREE "
                "reported as disagreeing rather than resolved silently."),
        },

        # ---------------------------------------------------- 4. THE VERDICTS
        "verdicts_per_cell": verdicts,
        "the_16_booleans_doc58_SS9_2": booleans,
        "directional_call_doc58_SS9_6": call,
        "bug_hunt_doc58_SS9_5": bug,
        "warmstart_spread_doc58_SS7_2": ws,

        # ---------------------------------------------------- 5. DESCRIPTIVE
        "reported_not_registered_doc58_SS9_7": reported,
        "mechanism_question_doc58_SS8_2": (
            "DOES A SUCCESSFUL RESCUE RESTORE THE q-ASYMMETRY UNDO SIGNATURE TO THE "
            "HEALTHY ARMS' PROFILE? Descriptive, NO boolean, NO criterion, and it "
            "may NOT be promoted onto a criterion after results exist (the exp44 "
            "no-promotion clause). The healthy profile is committed (doc 55 SS20.5): "
            "on correct q = 2 items the readout restores per-wire purity to its "
            "post-event-1 level (R2-A 0.8755 vs 0.8751); on q = 1 items it does not "
            "(0.8017 vs 0.8953). doc 56 SS15.2's amendment applies to I2: for the "
            "ENTANGLED arms the restoration target is the INTRO purity, not 1. "
            "Reported per arm, stratified by q and by correctness, in "
            "`reported_not_registered_doc58_SS9_7[<cell>].purity_doc58_SS8`, on "
            "every rescue arm AND both anchors."),

        # ---------------------------------------------------- 6. THE OUTCOME ROWS
        "outcome_rows_doc58_SS10": outcome_rows(booleans, verdicts, pcv,
                                                anchors_ok, unfloored),
        "scope_doc58_SS12": SCOPE,
        "what_this_could_not_resolve_doc58_SS19": COULD_NOT_RESOLVE,
        "I2_standing_doc58_SS18": (
            "doc 58's SS18 POST-REGISTRATION NOTE (2026-08-27). SS1.2's registered "
            "branch RESOLVES: the exp70 confirm draw REPRODUCED the failure that "
            "defines I2 (gap to the class-comparator 0.353 -> 0.335) while the "
            "`binds_N-in_T2` knife-edge flipped (0.6467 -> 0.6647 across the 0.65 "
            "bar). I2 STANDS EXACTLY AS REGISTERED -- the instance is the "
            "first-draw measurement on the frozen dataset, its baseline and bars "
            "unchanged -- and any quotation of the binds boolean carries the "
            "knife-edge note. No criterion moves."),
        "git_note": X.git_note(),
        "wall_time_sec": time.time() - t0,
    }

    path = X.fresh_results_path(RESULTS_BASE)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp72_aggregate] wrote %s" % path)

    if a.stage2:
        f = {"component": "exp72 stage-2 procedure-matched floors (doc 58 SS14)",
             "spec": X.SPEC, "spec_commit_line": X.spec_commit(),
             "platform": platform.platform(), "torch": torch.__version__,
             "floors": floors_stage2,
             "triggered_cells": triggered,
             "triggered_but_unfloored": unfloored,
             "severed_donors": {k: {"heldout_test_acc": r["heldout_test_acc"],
                                    "val_acc": r["val_acc"],
                                    "selected_epoch": r["selected_epoch"],
                                    "seed": r["seed"],
                                    "arm": r["arm"],
                                    "ckpt_sha256": r["ckpt_sha256"]}
                                for k, r in sorted(runs.items())
                                if r.get("kind") == "severed_donor"},
             "protocol": out["floors_doc58_SS14"]["severed_donor_protocol_doc58_SS14_4"],
             "rule3_note": ("written SEPARATELY from results_exp72.json because "
                            "stage 2 may run later than stage 1 and Rule 3 forbids "
                            "overwriting (doc 58 SS15.5)."),
             "git_note": X.git_note()}
        fp = X.fresh_results_path(FLOORS_BASE)
        json.dump(f, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[exp72_aggregate] wrote %s" % fp)

    # ---- console summary, in the registered reading order
    print("\n=== doc 58 SS5, ANCHORS (BLOCKING, READ FIRST) ===")
    if anchors:
        for i in ("I1", "I2"):
            p = anchors["per_instance"][i]
            print("  %s  max |delta| %.2e  bit-exact=%s  -> %s"
                  % (i, p["max_abs_per_seed_delta"], p["bit_exact"],
                     "PASS" if p["replicates"] else "FAIL"))
    print("  anchors_replicate_committed = %s" % anchors_ok)
    print("\n=== STAGE 1, per cell (mean +/- sd, 5 seeds) ===")
    for inst in ("I1", "I2"):
        print("  %s baseline %.4f  partial bar %.6f  full bar %.6f"
              % (inst, bar[inst]["baseline"], bar[inst]["rescue_partial_bar"],
                 bar[inst]["rescue_full_bar"]))
        for rescue in X.RESCUES:
            k = "%s|%s" % (inst, rescue)
            if k not in arms:
                print("      %-13s NOT RUN" % rescue)
                continue
            v = verdicts[k]
            print("      %-13s %.4f +/- %.4f   delta %+0.4f   trigger=%s  "
                  "full=%s partial=%s"
                  % (rescue, arms[k]["mean"], arms[k]["sd"],
                     v["mean_rescue_delta"], v["stage2_triggered"],
                     v["rescue_full"], v["rescue_partial"]))
    if pcv:
        print("\n=== doc 58 SS9.4, POSITIVE CONTROL ===")
        print("  R-budget on %s: %.4f +/- %.4f vs committed %.4f  "
              "(degradation %+0.4f)  degrades=%s"
              % (X.PC_ARM, pcv["exp72_mean"], pcv["exp72_sd"],
                 pcv["committed_mean"], -pcv["degradation"],
                 pcv["positive_control_degrades"]))
    print("\n=== doc 58 SS9.6, THE ONE DIRECTIONAL CALL ===")
    print("  rescue_partial(I1, R-warmstart): expected TRUE, outcome %s"
          % call["outcome"])
    if triggered:
        print("\n  stage-2 TRIGGERED cells: %s" % triggered)
    if unfloored:
        print("  TRIGGERED BUT UNFLOORED (booleans null, NEVER false): %s"
              % unfloored)
    return 0


def outcome_rows(booleans, verdicts, pcv, anchors_ok, unfloored):
    """doc 58 SS10 -- EVERY BRANCH WRITTEN IN ADVANCE. The applicable rows are
    emitted verbatim in spirit, with `applies` computed from the stored booleans.
    Nothing here is decided after the fact."""
    rf = {k: v for k, v in booleans.items() if k.startswith("rescue_")}
    known = [v for v in rf.values() if v is not None]
    any_rescue = any(v is True for v in rf.values())
    budget_rescues = any(booleans.get("rescue_%s(%s, R-budget)" % (b, i)) is True
                         for b in ("full", "partial") for i in ("I1", "I2"))
    ws_i1 = booleans.get("rescue_partial(I1, R-warmstart)")
    ws_i2 = booleans.get("rescue_partial(I2, R-warmstart)")
    lw = any(booleans.get("rescue_%s(%s, R-layerwise)" % (b, i)) is True
             for b in ("full", "partial") for i in ("I1", "I2"))
    idn = any(booleans.get("rescue_%s(%s, R-identity)" % (b, i)) is True
              for b in ("full", "partial") for i in ("I1", "I2"))
    contaminated = sorted(k for k, v in verdicts.items()
                          if v.get("floor_contaminated"))
    return [
        {"branch": "either anchor fails |delta| <= 0.002",
         "applies": bool(not anchors_ok),
         "registered_reading": (
             "THE RUN STOPS. The exp72 pipeline has drifted; SS5.2's floor-omission "
             "argument is void. Hunt the build; interpret nothing."),
         "may_not_be_claimed": "Nothing, from any arm."},
        {"branch": "R-budget rescues (either instance, full or partial)",
         "applies": bool(budget_rescues),
         "registered_reading": (
             "DEFLATIONARY AND IMPORTANT: 'UNDERTRAINED, NOT UNTRAINABLE.' The "
             "sophisticated fixes are unnecessary for that instance, doc 56 SS18.2's "
             "ledger sentence is amended IN PUBLIC, and the programme's four "
             "supply-without-conversion verdicts are re-read as statements about the "
             "committed 150-epoch recipe. The follow-up is doc 58 SS13 item 3's "
             "budget/schedule decomposition, and it becomes worth running."),
         "may_not_be_claimed": (
             "NOT that the earlier experiments were wrong -- they measured what they "
             "registered at the budget they registered. NOT a decomposition into "
             "budget vs schedule (SS6.1.1). NOT a claim about any instance not run "
             "here.")},
        {"branch": "rescue_partial(I1, R-warmstart) TRUE -- the registered call lands",
         "applies": bool(ws_i1 is True),
         "registered_reading": (
             "COMPOSITION IS A FINE-TUNING PROBLEM FOR THE PRODUCT REGISTER: the "
             "T = 1 solution is in the T = 2 basin's neighbourhood and standard "
             "training simply does not find it from a cold start. This would be the "
             "run's headline."),
         "may_not_be_claimed": (
             "MAY NOT BE CLAIMED ON THE FIRST DRAW ALONE -- the confirm draw at "
             "generator seed + 1000 with ALL ARMS OF I1 PRESENT, plus SS16.2's "
             "RETRAINED DONOR, is MANDATORY. Not a statement about DisCoCirc's "
             "expressive limits, and not about I2.")},
        {"branch": "warm-start rescues I1 but NOT I2",
         "applies": bool(ws_i1 is True and ws_i2 is False),
         "registered_reading": (
             "THE ENTANGLEMENT FAILURE IS DEEPER THAN COMPOSITION WARM-UP. Two "
             "encodings, same warm-start intervention, opposite outcomes => I2's "
             "failure is not merely 'the T = 2 basin is hard to reach from cold'. "
             "Reads onto doc 58 SS4.4's entanglement-induced-plateau mechanism and "
             "is exactly the branch doc 56 SS2 query 1 predicted would be different "
             "in kind."),
         "may_not_be_claimed": (
             "may NOT be reported as 'entanglement causes barren plateaus' -- one "
             "encoding, one gate, one inventory, 4 qubits.")},
        {"branch": "R-layerwise rescues while R-budget does not",
         "applies": bool(lw and not budget_rescues),
         "registered_reading": (
             "STAGED OPTIMISATION FINDS WHAT FULL-PARAMETER OPTIMISATION DOES NOT, "
             "and the raw 1.5x epoch excess is NOT the explanation because 3.0x did "
             "not rescue (SS6.3.2's guard, registered in advance)."),
         "may_not_be_claimed": (
             "not a result about Skolik et al.'s published circuit-GROWING "
             "procedure (SS4.2's adaptation clause).")},
        {"branch": "R-identity rescues while others do not",
         "applies": bool(idn and not budget_rescues and not lw
                         and ws_i1 is not True and ws_i2 is not True),
         "registered_reading": (
             "THE FAILURE IS AN INITIALISATION-BASIN FAILURE: the committed "
             "uniform-[0, 2pi) init lands the optimiser somewhere it cannot leave, "
             "and limiting effective depth at the first update is sufficient."),
         "may_not_be_claimed": (
             "not a claim that the instance was in a barren plateau -- this is a "
             "4-qubit circuit and no gradient-variance scaling was measured.")},
        {"branch": "NOTHING RESCUES (all 16 booleans FALSE, floors matched)",
         "applies": bool(anchors_ok and not any_rescue and len(known) == 16
                         and not unfloored),
         "registered_reading": (
             "THE WALLS ARE OPTIMISER-ROBUST AT THIS BUDGET CLASS. Four independent "
             "mitigations, one of them a 3x budget with a decay schedule, all fail "
             "on instances whose information content is CERTIFIED EXTRACTABLE. THIS "
             "STRENGTHENS THE TESTBED'S VALUE: exp72 becomes an instance the "
             "mitigation literature does not have (SS4.6), and the honest write-up "
             "sentence is 'these four operationalisations, at this budget class, on "
             "these two instances, did not convert certified-extractable "
             "information.'"),
         "may_not_be_claimed": (
             "MAY NOT BE REPORTED AS AN IMPOSSIBILITY. Not 'the circuit cannot learn "
             "this'; not 'trainability mitigations do not work'; not 'barren "
             "plateaus are unfixable'. Four procedures at one budget class on two "
             "instances is A RECORD, NOT A THEOREM, and doc 54 SS17.3's language "
             "binds.")},
        {"branch": "a rescue lands but the procedure-matched floor rises to meet it",
         "applies": bool(contaminated), "cells": contaminated,
         "registered_reading": (
             "THE PROCEDURE IS FITTING THE SPLIT, NOT THE LABELS. The rescue is NOT "
             "READ; the cell is reported as FLOOR-CONTAMINATED with both numbers."),
         "may_not_be_claimed": "no rescue claim of any kind from that cell."},
        {"branch": "positive_control_degrades TRUE",
         "applies": bool(pcv and pcv["positive_control_degrades"]),
         "registered_reading": (
             "the procedure DAMAGES a healthy arm. Every rescue under that procedure "
             "is reported WITH THE FLAG ATTACHED and read with suspicion (exp43c "
             "precedent, doc 24 SS5.2: a control that cannot fail is not a control)."),
         "may_not_be_claimed": (
             "a rescue under a flagged procedure may not be reported without the "
             "flag in the same sentence.")},
        {"branch": "rescues split across instances with no pattern",
         "applies": bool(any_rescue and not all(v is True for v in known)),
         "registered_reading": (
             "reported as NO PATTERN FOUND, with all 16 booleans and their per-seed "
             "values. At 5 seeds and 8 cells this is a genuinely likely outcome and "
             "is registered as an outcome, not as a failure of the experiment."),
         "may_not_be_claimed": (
             "no post-hoc narrative may be constructed over the pattern "
             "(SS9.8c: no ranking of the four procedures as a criterion).")},
    ]


SCOPE = [
    ("NO QUANTUM READING OF ANY KIND. The comparators are already at 1.0000 on "
     "both instances; there is nothing for a circuit to win. exp72 is about "
     "OPTIMISATION and only optimisation, and NO EXP72 NUMBER MAY APPEAR IN A "
     "SENTENCE CONTAINING 'ADVANTAGE'."),
    ("NOT A STATEMENT ABOUT BARREN PLATEAUS. These are 4-qubit circuits. No "
     "gradient-variance-versus-qubit-count scaling is measured, and doc 58 SS4.4 / "
     "SS4.5's asymptotic results are cited as MECHANISM LANGUAGE, never as "
     "diagnoses of these instances."),
    ("'THE FIELD'S MITIGATIONS' MEANS EXACTLY THESE FOUR OPERATIONALISATIONS, not "
     "the literature's every variant. Noted, not run: full barren-plateau-aware "
     "initialisation schemes (beta / Gaussian / reduced-domain / He / Xavier, and "
     "the CNN- and RL-based initialisers); quantum natural gradient; SGLD and other "
     "noisy/stochastic-gradient optimisers; encoding-entanglement reduction; "
     "cost-function locality; circuit-growing layerwise learning in Skolik et al.'s "
     "published form; the R-budget budget/schedule decomposition. A NULL ON THESE "
     "FOUR IS NOT A NULL ON THOSE."),
    ("One dataset, one task family, one category, one surface, two referents, two "
     "events, 24 nouns, simulation. doc 50 SS7's ceiling stands."),
    ("NOT 'MEANING', not composition-in-general, not DisCoCirc's expressive limits. "
     "A rescue relocates a failure from 'training' to 'training with a better "
     "recipe'; a null relocates nothing. Neither touches representability -- the "
     "bounds are 1.0000 in both cases and always were."),
    "Not novel-filler generalisation; exp69's held-out-PAIRING split, inherited.",
    "Not a statement about AraVec.",
    ("R-layerwise is an ADAPTATION of arXiv:2006.14904, not a replication (SS4.2). "
     "A null on it says NOTHING about their published circuit-growing method."),
]

COULD_NOT_RESOLVE = [
    ("The four procedures are not exhaustive and are not a random sample of the "
     "literature. exp72 runs one initialisation strategy, one staging strategy, one "
     "transfer strategy and one budget control."),
    ("R-layerwise is an adaptation, not a replication -- stated twice because it is "
     "the easiest sentence in the write-up to get wrong."),
    ("The one directional call rests on an EXISTENCE argument, not a precedent. "
     "This programme has never run a warm start; doc 38 / exp54's budget extension "
     "is the only in-programme rescue-shaped intervention and IT FAILED."),
    ("Stage 2's cost is conditional and could be zero. If no cell reaches its "
     "partial bar, no procedure-matched floor is ever run and doc 58's most careful "
     "section is never exercised."),
    ("The purity signature is DESCRIPTIVE and under-powered as a mechanism: with 5 "
     "seeds and two instances a restored q-asymmetry would be suggestive and "
     "nothing more."),
    ("What a null actually bounds is hard to state precisely. 'Optimiser-robust at "
     "this budget class' means four procedures, one of them 3x the budget with a "
     "decay schedule, on two instances, at 5 seeds, at n_test = 300. IT IS A "
     "RECORD. IT IS NOT A THEOREM."),
    ("doc 58 SS14.4's severed-donor protocol carried a registered ambiguity between "
     "cross-seed donor selection and seed linkage; the build resolved it in the only "
     "way consistent with SS15.3's run count and flagged the resolution in "
     "`floors_doc58_SS14`."),
]


if __name__ == "__main__":
    raise SystemExit(main())
