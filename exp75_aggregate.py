# -*- coding: utf-8 -*-
"""exp75_aggregate.py -- THE ONLY WRITER OF `results_exp75.json` (doc 62 SS20.4).

Cold-runnable. No shared state with the build session. It reads the 90 per-run
checkpoints in `exp75_ckpt/`, the comparator's own file, and the committed
`results_exp69.json`, and it writes `results_exp75_anchors.json` and
`results_exp75.json`. NO WORKER WRITES A RESULTS FILE, and Rule 3 forbids
overwriting any existing results file of record.

IT ENFORCES doc 62 SS18.4's GATING ORDER AND WILL NOT READ PAST A MISSING GATE:

    anchor  ->  reupload_engaged  ->  floor exists  ->  ONLY THEN
    `binds`, `reuploading_lifts/hurts` (with SS10.3's arbiter),
    `init_helps/hurts`  ->  `cc` and its ladder placement  ->  SS11's descriptives

AND doc 62 SS9.4's DISPOSITION, which is the whole content of E4's erratum:

  * 5/5 seeds engaged                 the arm implemented its intervention;
                                      whatever it measures is a FINDING
  * engaged, and no lift              SUPPLY PRESENT, MOVED UNDER PRESSURE, NOT
                                      CONVERTED -- the programme's fifth lever
                                      with the same shape, and the first where
                                      the supply was REPRESENTATIONALLY missing
  * < 5/5 seeds engaged               E4's OUTCOME SHAPE. The cell is
                                      INADMISSIBLE: its verdicts are stored as
                                      `null`, NEVER AS FALSE, and it is reported
                                      as "the intervention was available and
                                      unused", with the per-seed movements
  * 0/8 cells engaged                 NO RE-UPLOADING VERDICT AT ALL; the finding
                                      is about the optimiser's treatment of
                                      interleave blocks under this recipe, and
                                      the compute is reported as spent

`reuploading_lifts_T<t>` is NOT "re-uploading does not help" when it is FALSE --
that is the doc 46 SS17 sentence this whole rung exists to avoid repeating.

    python exp75_run.py --worker i --nworkers 5    # 90 runs, anchors first
    python exp75_comparator.py                     # tiers (i) and (ii)
    python exp75_aggregate.py                      # THIS FILE
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

import exp75_run as R

SPEC = R.SPEC
RESULTS_OUT = "results_exp75.json"
COMPARATOR = os.path.join(R.CKPT_DIR, "comparator.json")

BIND_BAR = 0.65                # doc 54 SS7's `anchor_real_category_binds`
FLOOR_MULTIPLE = 2.0           # doc 62 SS12, floor multiple 2 SD
DELTA_BAR = 0.06               # 2 x the binomial SD at n_test = 300 (sqrt(.25/300))
ENGAGEMENT_BAR = R.ENGAGEMENT_BAR                                  # 0.10 rad
BUG_HUNT_MARGIN = 0.06         # doc 62 SS15.1 -- STRUCTURALLY INERT here

# doc 56 SS18.1, verbatim -- the committed composition-fragility ladder.
COMMITTED_LADDER = [("angle (FA / EA-ctrl)", 0.004),
                    ("CZ-across (N-across)", 0.032),
                    ("eps-register (P-eps)", 0.132),
                    ("old-map register (F2 / A0-W)", 0.253),
                    ("entangled register (N-in)", 0.310)]

# doc 62 SS15 -- CONTEXT ONLY. NO CRITERION IN SS12 READS AGAINST ANY OF THESE.
RELAXATION_FLOOR = 0.9625


def sd(xs):
    xs = list(xs)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def load_runs(ckpt_dir, tag=""):
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
    """None-shaped when the cell is INCOMPLETE -- an incomplete cell has NO
    verdict and its booleans are stored as null, never as false."""
    rows = [runs[(arm, s)] for s in range(n_seeds) if (arm, s) in runs]
    if len(rows) < n_seeds:
        return {"complete": False, "n_runs": len(rows),
                "seeds_present": sorted(r["seed"] for r in rows),
                "n_seeds_registered": n_seeds,
                "note": ("INCOMPLETE -- doc 62 SS18.4: a cell without its full "
                         "seed set has NO verdict and its booleans are stored as "
                         "null, NEVER as false")}
    rows.sort(key=lambda r: r["seed"])
    acc = [r["heldout_test_acc"] for r in rows]
    cfg = R.ARM_CFG[R.base_arm(arm)]
    mv = [r.get("max_abs_dtheta_V") for r in rows]
    eng = [r.get("reupload_engaged") for r in rows]
    blk = {
        "complete": True, "arm": arm, "base_arm": R.base_arm(arm),
        "is_floor": bool(arm.startswith("S-shuffle")),
        "L": cfg["L"], "T": cfg["T"], "init": cfg["init"],
        "encoding": cfg["enc"], "kappa": 1.0, "family": cfg["family"],
        "role": cfg["role"],
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
        "init_scheme": rows[0]["init_scheme"].get("scheme"),
        "dataset_sha256": rows[0]["dataset_sha256"],
        "ckpt_sha256": [r["ckpt_sha256"] for r in rows],
        "wall_sec_total": float(sum(r["wall_sec"] for r in rows)),
        # ---- doc 62 SS9.3 / SS11.4 --------------------------------------
        "max_abs_dtheta_V_per_seed": mv,
        "max_abs_dtheta_V_final_per_seed": [r.get("max_abs_dtheta_V_final")
                                            for r in rows],
        "max_abs_dtheta_verb_per_seed": [r.get("max_abs_dtheta_verb")
                                         for r in rows],
        "engaged_per_seed": eng,
        "n_seeds_engaged": (None if cfg["L"] == 1
                            else sum(1 for x in eng if x is True)),
    }
    blk.update(descriptives(rows))
    blk["harmonic_budget_doc62_SS11_3"] = harmonic_summary(rows, cfg["L"])
    return blk


def descriptives(rows):
    """doc 62 SS12.7 -- REPORTED, NOT REGISTERED, and NONE MAY BE PROMOTED ONTO A
    CRITERION AFTER THE FACT (the exp44 no-promotion clause)."""
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
            "purity_traces_doc62_SS11_1": purity_summary(rows),
            "reported_not_registered_doc62_SS12_7": (
                "per-q held-out accuracy; per-surface-order accuracy; the "
                "val-minus-held-out selection gap per seed; selected epoch per "
                "seed; per-verb-pair accuracy; SS11's traces and harmonic "
                "budgets; per-seed deltas; tier (i) and tier (ii) comparator "
                "values; `cc` for all four cells; the L = 2 vs L = 3 ordering. NO "
                "VERDICT READS ANY OF THESE AND NONE MAY BE PROMOTED ONTO A "
                "CRITERION AFTER RESULTS EXIST.")}


def purity_summary(rows):
    """doc 62 SS11.1 / SS11.2 -- the per-wire trace, stratified by q and by
    correctness. DESCRIPTIVE. No criterion reads it (SS11.3's no-promotion clause
    covers it)."""
    acc = {}
    for r in rows:
        for it in r["heldout_items"]:
            pur = it.get("purity")
            if not pur:
                continue
            key = ("q%d" % it["query_index"],
                   "correct" if it["correct"] else "wrong")
            for stage, vals in pur.items():
                acc.setdefault((key, stage), []).extend(vals)
    out = {}
    for (key, stage), vals in acc.items():
        out.setdefault("%s/%s" % key, {})[stage] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n": len(vals)}
    return {
        "by_q_and_correctness": out,
        "intro_purity_is_1_by_construction_doc62_SS7_5": (
            "only single-qubit gates act on individual wires in the exp75 intro, "
            "so the referents are still introduced as a PRODUCT STATE and the "
            "per-wire intro purity is 1.0 by construction (asserted at smoke in "
            "complex128). THE THREE LATER STAGES CARRY THE STORY, and doc 56's "
            "concurrence machinery has nothing to gate here."),
        "registered_question_doc62_SS11_2": (
            "does adding uploads change the q-asymmetry UNDO SIGNATURE, and does "
            "an arm that lifts (if any does) show a restored or a degraded one? "
            "THIS RUNG IS THE ONE WHERE THE COMPARISON IS DIRECT: doc 55 SS20.5's "
            "healthy profile was measured ON R2-A ITSELF -- same family, same "
            "wires, same readout -- so doc 56 SS15.2's transposition and doc 61 "
            "SS9.4's re-basing are NOT needed. Committed profile: on correct "
            "q = 2 items the readout restores per-wire purity to its "
            "post-event-1 level (0.8755 vs 0.8751); on q = 1 items it does not "
            "(0.8017 vs 0.8953)."),
        "no_promotion": (
            "DESCRIPTIVE. NO BOOLEAN. NO CRITERION READS IT, AND IT MAY NOT BE "
            "PROMOTED AFTER RESULTS EXIST (doc 62 SS12.7, the exp44 clause)."),
    }


def harmonic_summary(rows, L):
    """doc 62 SS11.3 -- THE DESCRIPTIVE E4 COULD NEVER PRODUCE.

    DOES A TRAINED RE-UPLOADING MODEL USE THE HARMONICS IT WAS GIVEN? This is the
    measurement that makes a null interpretable: a cell that ENGAGED (SS9.3) and
    lifted nothing with e_{k>=2} near zero is 'THE OPTIMISER DECLINED THE
    SUPPLY'; the same cell with substantial e_{k>=2} is 'THE SUPPLY WAS USED AND
    DID NOT HELP'. DESCRIPTIVE, NO BOOLEAN, NO CRITERION, NO PROMOTION."""
    if L == 1:
        return None
    per_seed, ks = [], [str(k) for k in range(1, L + 1)]
    for r in rows:
        b = r.get("harmonic_budget_doc62_SS11_3")
        per_seed.append(None if not b else b["normalised_energy_mean"])
    mean = {}
    for k in ks:
        vals = [p[k] for p in per_seed if p and p.get(k) is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    return {"harmonics": ks, "normalised_energy_mean_over_seeds": mean,
            "normalised_energy_per_seed": per_seed,
            "instrument": ("the SS8.2 sweep re-run on each TRAINED checkpoint: a "
                           "48-point sweep of one referent's intro angle over "
                           "[0, 2pi) on 4 held-out stories, fitted in the "
                           "(2L+1)-term basis, normalised energy per harmonic "
                           "e_k = (a_k^2 + b_k^2) / sum_j (a_j^2 + b_j^2)"),
            "status_doc62_SS11_3": ("DESCRIPTIVE. NO BOOLEAN. NO CRITERION READS "
                                    "IT AND IT MAY NOT BE PROMOTED.")}


def floor_clearance(arm_blk, floor_blk):
    """doc 62 SS12 with doc 55 SS6.6's THREE SD READINGS. THE FLOOR'S OWN SD IS
    PRIMARY; all three are computed and reported, and DISAGREEMENT IS REPORTED AS
    DISAGREEMENT RATHER THAN RESOLVED SILENTLY (doc 60's ruling 5,
    PRIMARY-DECIDES)."""
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
    contaminated = bool(f >= BIND_BAR or (a - f) < DELTA_BAR)
    return {
        "arm_mean": a, "floor_mean": f, "delta": a - f,
        "readings": read, "primary": "floor_sd_PRIMARY",
        "clears_floor": read["floor_sd_PRIMARY"]["clears"],
        "readings_agree": bool(agree),
        "disagreement_note": (None if agree else
                              "doc 55 SS6.6: the three SD readings DISAGREE and "
                              "that is REPORTED AS DISAGREEMENT rather than "
                              "resolved silently. The floor's own SD is primary."),
        "floor_contaminated_doc62_SS18_5": contaminated,
        "contamination_rule": (
            "doc 62 SS18.5: IF A FLOOR RISES TO MEET ITS ARM the cell is reported "
            "as FLOOR-CONTAMINATED with BOTH numbers and IS NOT READ (doc 58 "
            "SS10, doc 61 SS15.5). Operationalised transparently as floor_mean "
            ">= %.2f (the floor itself reaches the binding bar) OR arm_mean - "
            "floor_mean < %.2f (the separation is inside the registered "
            "resolution). At +6/+12 parameters over 120 this is a LOW BUT "
            "NON-ZERO risk and is registered rather than assumed away."
            % (BIND_BAR, DELTA_BAR)),
        "floor_construction_doc62_SS18_1": (
            "the committed S-shuffle (`exp69_run.relabel_shuffle_q`, IMPORTED "
            "unchanged) -- bearer designation reassigned PER PAIR, independently "
            "of the features -- same items, same splits, same 5 seeds, run under "
            "the IDENTICAL ARM PROCEDURE. **BOTH L AND THE INIT ARE PART OF THE "
            "PROCEDURE, so EIGHT CELLS GET EIGHT FLOORS** (doc 58 SS14's "
            "argument, inherited: a floor measured under one procedure does not "
            "bound what another can achieve on severed labels). All eight ran in "
            "STAGE 1 (doc 62 SS18.2, orchestrator ruling 1), because the trigger "
            "is satisfied by arithmetic: the committed family binds at "
            "0.9187-0.9227, so every non-catastrophic cell clears 0.65."),
    }


def admissible(cell, engagement, floors_ok):
    """doc 62 SS18.4's GATING ORDER, made mechanical. A cell is read ONLY when it
    is complete, ENGAGED ON 5/5 SEEDS, has a complete matched floor, and that
    floor is not contaminated. Anything else yields `null`, NEVER `False`."""
    if not cell.get("complete"):
        return False, "the cell is incomplete -- no verdict (doc 62 SS18.4)"
    if engagement is None or not engagement.get("reupload_engaged"):
        return False, ("the cell is NOT ENGAGED on 5/5 seeds -- E4's OUTCOME "
                       "SHAPE. INADMISSIBLE: verdicts are null, NEVER FALSE, and "
                       "the cell is reported as 'the intervention was available "
                       "and unused' (doc 62 SS9.4)")
    if floors_ok is None:
        return False, ("the cell's procedure-matched floor has not been run -- NO "
                       "VERDICT IS READ PAST A MISSING FLOOR (doc 62 SS18.4)")
    if floors_ok.get("floor_contaminated_doc62_SS18_5"):
        return False, ("the cell is FLOOR-CONTAMINATED: the floor rose to meet "
                       "its arm. Reported with both numbers and NOT READ "
                       "(doc 62 SS18.5, SS14)")
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=RESULTS_OUT)
    ap.add_argument("--anchor-out", default=R.ANCHOR_OUT)
    ap.add_argument("--comparator", default=None)
    a = ap.parse_args()
    t0 = time.time()
    R.CKPT_DIR = a.ckpt_dir
    comp_path = a.comparator or os.path.join(a.ckpt_dir, "comparator.json")

    store = R.load_part()
    runs = load_runs(a.ckpt_dir, tag=a.tag)
    print("[exp75-agg] %d per-run checkpoints in %s" % (len(runs), a.ckpt_dir))

    head = {
        "experiment": ("exp75 -- THE RE-UPLOADING RUNG (doc 62, D2 of doc 59's "
                       "six-direction plan). E4's REPAIR."),
        "spec": SPEC,
        "question": ("if a referent's single scalar angle is uploaded L times "
                     "instead of once, with trainable processing between the "
                     "uploads -- the field's canonical construction for making a "
                     "small circuit expressive -- does the added frequency supply "
                     "convert into trained held-out accuracy on this programme's "
                     "composition task? (doc 62 SS0.1)"),
        "orchestrator_rulings": {
            "1_run_count": ("90 RUNS ACCEPTED: 10 anchor + 40 trained cells + 40 "
                            "matched floors, all eight floors in stage 1 per the "
                            "exp73 precedent (doc 62 SS18.2, SS19.1)."),
            "2_kappa": ("kappa = 1 FIXED AND NOT A TRAINABLE PARAMETER, RATIFIED "
                        "with doc 62 SS4.2's four-point rationale. Validity "
                        "becomes a property of the BUILD, not of the optimiser's "
                        "output -- which is E4 exactly."),
            "3_engagement_bar": ("0.10 rad ACCEPTED as calibrated: it fails ALL "
                                 "34 E4 runs, including the most-moved (+0.0780)."),
            "4_relaxation_bound": ("NOT computed as pre-work; registered "
                                   "COMPUTABLE-ON-DEMAND. Its nesting argument "
                                   "already proves it monotone >= 0.9625 and "
                                   "makes the bug-hunt gate INERT, which is the "
                                   "only use doc 62 has for it (SS2.4, SS15.1)."),
            "5_directional_call": ("the declined call is RATIFIED. exp75 places "
                                   "NO DIRECTIONAL CALL (doc 62 SS13); the "
                                   "candidate `reuploading_lifts_T2 = FALSE` was "
                                   "declined as SAFE BY CONSTRUCTION rather than "
                                   "falsifiable (SS21 item 5)."),
            "6_record": ("directional-call record: ONE held twice-drawn (exp70), "
                         "ONE missed (exp72 warm-start), ONE pending (exp73 "
                         "amplitude_hurts_T1)."),
            "7_lit_ordinal": "doc 62 is the THIRTEENTH literature check.",
        },
        "aggregated_from": a.ckpt_dir, "aggregator": os.path.basename(__file__),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "dataset": store[2], "dataset_sha256": store[1]["dataset_sha256"],
        "smoke_results": R.SMOKE_OUT, "anchor_results": a.anchor_out,
        "n_seeds": a.seeds, "sd_convention": "sample sd, ddof=1",
        "training": {"lr": R.LR, "batch": R.BATCH, "max_epochs": R.MAX_EPOCHS,
                     "rng_base": R.RNG_BASE, "kappa": 1.0,
                     "selection": "in-distribution val accuracy",
                     "heldout": "read ONCE from the selected checkpoint"},
        "architecture_doc62_SS7": {
            "intro": ("U_L(theta) = RY(theta) . V_{L-1} . RY(theta) . ... . V_1 . "
                      "RY(theta) per referent wire, EVERY upload carrying the "
                      "SAME angle theta (kappa = 1 FIXED)"),
            "V_j": ("Circuit4(1, 3) -- the COMMITTED ansatz at one qubit, RX.RZ "
                    "three times, 6 parameters, CRX cascade EMPTY by "
                    "construction. NO NEW ANSATZ (doc 55 SS4.7)."),
            "sharing": ("ONE parameter set per layer index j, applied to EVERY "
                        "referent wire (doc 62 SS7.2): the wire index is an "
                        "arbitrary compiler-assigned referent slot, not a "
                        "semantic role."),
            "parameters": {"committed_FA": 120, "RU-L2": 126, "RU-L3": 132,
                           "vs_committed": {"RU-L2": "+5.0 %",
                                            "RU-L3": "+10.0 %"}},
            "everything_else": ("the committed recipe UNCHANGED (doc 62 SS6.2): "
                                "same dataset, splits, FA angle features, six "
                                "verb blocks, readout, 150 epochs, lr 0.02, Adam, "
                                "batch 8, selection rule, n_test = 300. THE ONLY "
                                "CHANGE IS THE INTRO."),
        },
        "advantage_prohibition": (
            "NO EXP75 NUMBER MAY APPEAR IN A SENTENCE CONTAINING 'ADVANTAGE' "
            "(doc 52 SS1; doc 56 SS0.1; doc 58 SS12; doc 61 SS0.3; doc 62 SS0.3 / "
            "SS16). The model's output is a degree-L TRIGONOMETRIC POLYNOMIAL in "
            "two scalars and the function-class reference is a LINEAR MODEL on "
            "the joint harmonic features -- 25 numbers at L = 2, 49 at L = 3."),
        "scope_doc62_SS16": [
            "NO QUANTUM READING OF ANY KIND.",
            ("NOT A STATEMENT ABOUT DATA RE-UPLOADING IN GENERAL: exp75 runs "
             "FIXED UNIT PREFACTORS, L in {2, 3}, one-qubit interleave blocks "
             "from the committed ansatz, on ONE scalar feature per referent, two "
             "referents, two events, one category, one surface, 24 nouns, in "
             "simulation. The TRAINABLE-PREFACTOR variant is explicitly DECLINED "
             "(SS4.3, SS17 items 1-2) and no exp75 outcome speaks to it."),
            ("NOT A STATEMENT ABOUT UNIVERSALITY: universality is an ASYMPTOTIC "
             "statement about arbitrarily many uploads and arbitrary trainable "
             "processing; L <= 3 with 6-12 interleave parameters is not that, and "
             "no null here contradicts Perez-Salinas et al. or Schuld et al."),
            "NOT A STATEMENT ABOUT BARREN PLATEAUS -- 2-wire circuits.",
            ("NOT A CLAIM ABOUT THE DEGREE-L FUNCTION CLASS even if a cell beats "
             "tier (ii): the comparator is a SUPERSET IN COEFFICIENT SPACE and a "
             "RESTRICTION ACROSS CONDITIONING GROUPS (SS10.4), and both "
             "qualifiers travel together."),
            ("NOT A REPAIR OF E4's VERDICT: E4 produced NO verdict. Nothing here "
             "retro-validates `results_exp60.json`'s `reuploading_relieves = "
             "FALSE`, which remains INADMISSIBLE (doc 46 SS10.5)."),
            "ONE DATASET, ONE TASK FAMILY, TWO REFERENTS, TWO EVENTS, 5 SEEDS.",
        ],
    }

    # ------------------------------------------------------------------ STEP 1
    # doc 62 SS12.1 / SS18.4 -- THE ANCHOR, READ FIRST AND BLOCKING.
    anchor, anchor_err = None, None
    try:
        anchor = R.anchor_report(store)
        json.dump(anchor, open(a.anchor_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[exp75-agg] wrote %s -- anchor_replicates_committed=%s "
              "(bit_exact_all=%s)"
              % (a.anchor_out, anchor["anchor_replicates_committed"],
                 anchor["bit_exact_all"]))
    except (RuntimeError, AssertionError, KeyError, OSError) as exc:
        anchor_err = str(exc)
        print("[exp75-agg] ANCHOR NOT AVAILABLE: %s" % anchor_err)
    anchor_ok = bool(anchor and anchor["anchor_replicates_committed"])

    # ------------------------------------------------------------------ STEP 2
    cells = {arm: arm_block(runs, arm, a.seeds) for arm in R.TRAINED_CELLS}
    floors = {arm: arm_block(runs, arm, a.seeds) for arm in R.FLOOR_ARMS}
    anchors = {arm: arm_block(runs, arm, a.seeds) for arm in R.ANCHOR_ARMS}

    # doc 62 SS12.2 / SS9.3 -- `reupload_engaged_<cell>`, READ SECOND.
    engagement = {}
    for arm, blk in cells.items():
        if not blk.get("complete"):
            engagement[arm] = None
            continue
        n_eng = blk["n_seeds_engaged"]
        engagement[arm] = {
            "reupload_engaged": bool(n_eng == a.seeds),
            "n_seeds_engaged": n_eng, "n_seeds": a.seeds,
            "bar_rad": ENGAGEMENT_BAR,
            "max_abs_dtheta_V_per_seed": blk["max_abs_dtheta_V_per_seed"],
            "max_abs_dtheta_verb_per_seed": blk["max_abs_dtheta_verb_per_seed"],
            "definition": ("max_j max |theta_V(selected) - theta_V(init)| >= "
                           "%.2f rad, ABSOLUTE, PER RUN, on the maximum over the "
                           "block's 6 or 12 parameters -- NOT a mean over seeds. "
                           "E4's arm passed on its mean only because 34 runs were "
                           "averaged into a summary." % ENGAGEMENT_BAR),
            "disposition_doc62_SS9_4": (
                "5/5 ENGAGED -- the arm implemented its intervention and whatever "
                "it measures is a FINDING about re-uploading."
                if n_eng == a.seeds else
                "< 5/5 ENGAGED -- E4's OUTCOME SHAPE. THE CELL IS INADMISSIBLE: "
                "its SS12.3-SS12.5 verdicts are stored as null, NEVER as FALSE, "
                "and it is reported as 'THE INTERVENTION WAS AVAILABLE AND "
                "UNUSED', with the per-seed movements -- exactly as doc 46 "
                "SS10.5 required and doc 46 SS17 delivered too late."),
        }
    n_cells_engaged = sum(1 for v in engagement.values()
                          if v and v["reupload_engaged"])

    # ------------------------------------------------------------------ STEP 3
    clearance = {arm: floor_clearance(cells[arm], floors.get(R.FLOOR_OF[arm]))
                 for arm in R.TRAINED_CELLS}
    adm = {}
    for arm in R.TRAINED_CELLS:
        ok, why = admissible(cells[arm], engagement.get(arm), clearance.get(arm))
        adm[arm] = {"admissible": ok, "reason": why}

    # ------------------------------------------------------------------ STEP 4
    # doc 62 SS12.3 -- `binds_<cell>`, 8 BOOLEANS, CONJUNCTIVE.
    binds = {}
    for arm in R.TRAINED_CELLS:
        blk, cl = cells[arm], clearance.get(arm)
        if not adm[arm]["admissible"]:
            binds["binds_%s" % arm] = {"value": None, "reason": adm[arm]["reason"]}
            continue
        part1 = bool(blk["mean"] >= BIND_BAR)
        part2 = bool(cl["clears_floor"])
        binds["binds_%s" % arm] = {
            "value": bool(part1 and part2),
            "rule": ("binds = (heldout_mean >= %.2f) AND (heldout_mean >= "
                     "floor_mean + %.1f * floor_sd) -- CONJUNCTIVE (doc 55 SS6.3, "
                     "doc 56 SS6.3, doc 61 SS10.2, doc 62 SS12.3)"
                     % (BIND_BAR, FLOOR_MULTIPLE)),
            "part_1_absolute_bar": {"bar": BIND_BAR, "mean": blk["mean"],
                                    "passes": part1},
            "part_2_floor": cl,
            "registered_expectation_doc62_SS12_3": (
                "stated as ARITHMETIC, not as a call: the committed family binds "
                "at 0.9187-0.9227, so any non-catastrophic cell clears 0.65. "
                "`binds` HERE IS A SANITY GATE, NOT THE HEADLINE, and SS18.2 uses "
                "that fact to justify running all eight floors up front."),
        }

    # ------------------------------------------------------------------ STEP 5
    # doc 62 SS12.4 -- THE HEADLINE, 4 BOOLEANS, TWO-SIDED.
    committed = {}
    src69 = json.load(open(R.EXP69_RESULTS, encoding="utf-8"))
    for anchor_arm, source in R.ANCHOR_SOURCE.items():
        committed[R.ANCHOR_T[anchor_arm]] = {
            "arm": source, "mean": float(src69["arms"][source]["mean"]),
            "sd": float(src69["arms"][source]["sd"]),
            "per_seed": list(src69["arms"][source]["per_seed"]),
            "source": R.EXP69_RESULTS}
    comparator = None
    if os.path.exists(comp_path):
        comparator = json.load(open(comp_path, encoding="utf-8"))

    lifts = {}
    for T in (1, 2):
        pool = [arm for arm in R.TRAINED_CELLS
                if R.ARM_CFG[arm]["T"] == T and adm[arm]["admissible"]]
        base = committed[T]["mean"]
        if not pool:
            for name in ("reuploading_lifts_T%d" % T, "reuploading_hurts_T%d" % T):
                lifts[name] = {
                    "value": None,
                    "reason": ("NO ADMISSIBLE CELL AT T = %d. doc 62 SS18.4's "
                               "gating order forbids reading a gain criterion "
                               "over cells that are disengaged, floor-less, "
                               "floor-contaminated or incomplete, so the boolean "
                               "is null and NOT FALSE -- and null here IS NOT "
                               "'re-uploading does not help', which is the doc 46 "
                               "SS17 sentence this rung exists to avoid "
                               "repeating (doc 62 SS9.4, SS14)." % T),
                    "cells_considered": [],
                    "inadmissible": {arm: adm[arm]["reason"]
                                     for arm in R.TRAINED_CELLS
                                     if R.ARM_CFG[arm]["T"] == T}}
            continue
        best = max(pool, key=lambda x: cells[x]["mean"])
        delta = cells[best]["mean"] - base
        arb = arbiter(comparator, T, best, delta)
        common = {
            "committed_baseline": committed[T],
            "best_admissible_cell": best,
            "best_admissible_cell_mean": cells[best]["mean"],
            "delta": delta,
            "cells_considered": sorted(pool),
            "per_cell_delta": {arm: cells[arm]["mean"] - base for arm in pool},
            "inadmissible": {arm: adm[arm]["reason"] for arm in R.TRAINED_CELLS
                             if R.ARM_CFG[arm]["T"] == T
                             and not adm[arm]["admissible"]},
            "admissibility_note_doc62_SS18_4": (
                "delta(T) is the MAXIMUM OVER ADMISSIBLE CELLS ONLY. doc 62 "
                "SS12.4's formula reads 'max over {L} x {init}'; SS18.4's gating "
                "order and SS9.4's null-not-FALSE disposition together mean a "
                "disengaged, floor-less or floor-contaminated cell has NO verdict "
                "and therefore cannot supply the maximum. Every excluded cell is "
                "listed above with its reason."),
            "comparator_arbiter_doc62_SS10_3": arb,
        }
        lifts["reuploading_lifts_T%d" % T] = dict(common, **{
            "value": bool(delta >= DELTA_BAR),
            "bar": base + DELTA_BAR,
            "rule": "delta(T) >= +%.2f" % DELTA_BAR,
            "arbiter_binds_doc62_SS12_4_1": (
                "SS10.3's second row is NOT OPTIONAL: a lift that EXCEEDS the "
                "information gap is a BASELINE-TRAINABILITY ARTEFACT -- the "
                "committed single-encode arm under-trained its own class (SS1.4 "
                "measured 0.04 of unconverted supply) -- and MAY NOT BE CLAIMED "
                "AS A RE-UPLOADING RESULT."),
            "confirm_draw_owed_doc62_SS20_2": bool(delta >= DELTA_BAR),
        })
        lifts["reuploading_hurts_T%d" % T] = dict(common, **{
            "value": bool(delta <= -DELTA_BAR),
            "bar": base - DELTA_BAR,
            "rule": "delta(T) <= -%.2f" % DELTA_BAR,
            "reading_if_true_doc62_SS14": (
                "THE INTERVENTION DEGRADES A HEALTHY ARM: more frequencies made "
                "the OPTIMISATION worse at fixed everything-else. Reads onto doc "
                "62 SS5.3's Fourier-locking mechanism (the non-linear coupling of "
                "encoding weights and entangling gates) and onto arXiv:2008.08605's "
                "'scaling of data reduces the trainability'. NOT a claim that the "
                "CLASS is worse -- the degree-L span STRICTLY CONTAINS the "
                "baseline's (SS2.4), so a loss is an OPTIMISATION statement by "
                "construction."),
        })
    for T in (1, 2):
        lp = lifts.get("reuploading_lifts_T%d" % T, {}).get("value")
        hp = lifts.get("reuploading_hurts_T%d" % T, {}).get("value")
        assert not (lp and hp), "both lifts and hurts TRUE at T=%d" % T

    # ------------------------------------------------------------------ STEP 6
    # doc 62 SS12.5 -- `init_helps` / `init_hurts`, 8 BOOLEANS.
    init_bools = {}
    for L in R.UPLOADS:
        for T in (1, 2):
            u = "RU-L%d-T%d-uniform" % (L, T)
            i = "RU-L%d-T%d-identity" % (L, T)
            key = "L%d_T%d" % (L, T)
            if not (adm[u]["admissible"] and adm[i]["admissible"]):
                for nm in ("init_helps_%s" % key, "init_hurts_%s" % key):
                    init_bools[nm] = {
                        "value": None,
                        "reason": ("one or both cells are inadmissible: uniform "
                                   "-> %s ; identity -> %s (doc 62 SS18.4)"
                                   % (adm[u]["reason"], adm[i]["reason"]))}
                continue
            d = cells[i]["mean"] - cells[u]["mean"]
            paired = [x - y for x, y in zip(cells[i]["per_seed"],
                                            cells[u]["per_seed"])]
            common = {"init_delta_identity_minus_uniform": d,
                      "uniform_mean": cells[u]["mean"],
                      "identity_mean": cells[i]["mean"],
                      "paired_by_seed_deltas_REPORTED": paired,
                      "booleans_read_the_means": True,
                      "interpretation_binding_doc62_SS3_5": (
                          "ON THIS RUNG THE INIT CHANGES THE SPECTRAL CONTENT OF "
                          "THE INITIAL MODEL, not the effective depth at the "
                          "first update: identity init REMOVES frequency supply "
                          "at t = 0 (at V = 0 the circuit IS RY(L.theta), "
                          "spectrum {0, +-L}) while uniform init supplies it. "
                          "THESE BOOLEANS ARE THEREFORE NOT COMPARABLE WITH doc "
                          "58 SS19.1's OR doc 61 SS10.4's, AND SS16 / SS12.8(e) "
                          "FORBID THE SUBTRACTION. An init split here is the "
                          "PREDICTED asymmetry and is read against SS3.")}
            init_bools["init_helps_%s" % key] = dict(
                common, value=bool(d >= DELTA_BAR), bar=DELTA_BAR,
                rule="init_delta >= +%.2f" % DELTA_BAR)
            init_bools["init_hurts_%s" % key] = dict(
                common, value=bool(d <= -DELTA_BAR), bar=-DELTA_BAR,
                rule="init_delta <= -%.2f" % DELTA_BAR)

    # ------------------------------------------------------------------ STEP 7
    # doc 62 SS12.6 -- COMPOSITION COST, DESCRIPTIVE, NO BOOLEAN, NO CALL.
    cc = {}
    for L in R.UPLOADS:
        for init in R.INITS:
            t1 = "RU-L%d-T1-%s" % (L, init)
            t2 = "RU-L%d-T2-%s" % (L, init)
            key = "L%d_%s" % (L, init)
            if not (cells[t1].get("complete") and cells[t2].get("complete")):
                cc[key] = None
                continue
            v = cells[t1]["mean"] - cells[t2]["mean"]
            lo = min(x for _n, x in COMMITTED_LADDER)
            cc[key] = {
                "cc": v, "T1_mean": cells[t1]["mean"], "T2_mean": cells[t2]["mean"],
                "committed_angle_rung": 0.004,
                "ladder_placement": ("B2 -- the angle rung's 0.004-0.032 band"
                                     if -DELTA_BAR <= v <= 0.032 else
                                     ("B1 -- risen above the angle rung"
                                      if v > 0.032 else
                                      "UNREGISTERED LADDER PLACEMENT")),
                "both_cells_admissible": bool(adm[t1]["admissible"]
                                              and adm[t2]["admissible"]),
                "ladder_min": lo,
            }
    cc_meta = {
        "definition": "cc(L, init) = heldout_mean(T=1) - heldout_mean(T=2)",
        "committed_ladder_doc56_SS18_1": {n: v for n, v in COMMITTED_LADDER},
        "registered_question_doc62_SS12_6": (
            "DOES ADDING FREQUENCY SUPPLY TO THE FAMILY WHOSE COMPOSITION IS FREE "
            "DESTABILISE IT? BOTH BRANCHES ARE REGISTERED AND NEITHER IS CALLED. "
            "B1 -- cost rises with encoding complexity: the ladder's own ordering "
            "is by encoding complexity and re-uploading is a strictly richer "
            "encoding on the same wires, so cc rises above 0.004, plausibly "
            "toward the eps-register's 0.132, and rises with L. B2 -- cost is "
            "about the MAP's geometry, not its degree: the ladder's one "
            "CONTROLLED movement is dcc_map = -0.1207 (the eps-map HALVED the "
            "register's collapse at bit-identical architecture), i.e. collapse "
            "tracked DEAD ZONES and RANGE COMPRESSION, neither of which "
            "re-uploading introduces -- the map stays one angle per referent, "
            "dense and unclipped -- so cc stays at 0.004-0.032 for both L."),
        "status": ("DESCRIPTIVE, NO BOOLEAN, NO CALL. If cc lands outside both "
                   "branches it is reported as UNREGISTERED LADDER PLACEMENT with "
                   "both branches quoted and neither claimed; the ladder gains a "
                   "point with a QUESTION MARK, not a story, and no post-hoc "
                   "mechanism may be constructed over it (doc 62 SS12.7, SS14)."),
    }

    # ------------------------------------------------------------------ STEP 8
    # doc 62 SS15.1 -- `bug_hunt_triggered`, STRUCTURALLY INERT, stored only.
    bug_hunt = {}
    for arm in R.TRAINED_CELLS:
        if not cells[arm].get("complete"):
            bug_hunt[arm] = None
            continue
        bug_hunt[arm] = {
            "relaxed_bound_floor": RELAXATION_FLOOR,
            "measured": cells[arm]["mean"],
            "trigger_threshold_if_bound_were_the_floor":
                RELAXATION_FLOOR + BUG_HUNT_MARGIN,
            "bug_hunt_triggered": bool(
                cells[arm]["mean"] > RELAXATION_FLOOR + BUG_HUNT_MARGIN),
        }
    bug_hunt_meta = (
        "doc 62 SS2.4 / SS15.1: THE L = 1 SPAN IS A SUBSPACE OF THE L = 2 SPAN, "
        "WHICH IS A SUBSPACE OF THE L = 3 SPAN, so the relaxed bound is "
        "NON-DECREASING IN L and is therefore >= 0.9625 for every exp75 arm. The "
        "programme's standard bug-hunt gate fires above bound + 0.06, i.e. above "
        ">= 1.0225, WHICH IS UNREACHABLE. `bug_hunt_triggered` IS STRUCTURALLY "
        "INERT HERE, is computed and stored for completeness, and NO READING "
        "DEPENDS ON IT (doc 58 SS9.5, doc 61 SS12.1, third use). Orchestrator "
        "ruling 4 registered the relaxation itself as COMPUTABLE-ON-DEMAND and "
        "NOT COMPUTED, because the monotonicity argument is the only use doc 62 "
        "has for it.")

    out = dict(head)
    out.update({
        "gating_order_doc62_SS18_4": (
            "anchor -> reupload_engaged -> floor exists -> ONLY THEN binds, "
            "reuploading_lifts/hurts (with SS10.3's arbiter), init_helps/hurts "
            "-> cc -> SS11's descriptives. A CELL WHOSE FLOOR HAS NOT BEEN RUN OR "
            "WHICH IS NOT ENGAGED HAS NO VERDICT AND ITS BOOLEANS ARE STORED AS "
            "null, NEVER AS FALSE."),
        "n_runs_found": len(runs),
        "n_runs_registered": len(R.ALL_ARMS) * a.seeds,
        "anchor_gate_doc62_SS12_1": {
            "anchor_replicates_committed": (None if anchor is None
                                            else anchor["anchor_replicates_committed"]),
            "bit_exact_all": (None if anchor is None else anchor["bit_exact_all"]),
            "error": anchor_err, "file": a.anchor_out,
            "blocking": True,
            "consequence_if_failed": (
                "doc 62 SS6.3 / SS14 row 1: THE RUN STOPS. The pipeline has "
                "drifted, SS6.3.2's floor-omission argument is VOID, and NOTHING "
                "ELSE IN THIS FILE MAY BE READ. Hunt the build; interpret "
                "nothing."),
        },
        "anchors": anchors,
        "arms": cells, "floors": floors,
        "engagement_doc62_SS12_2": engagement,
        "n_cells_engaged": n_cells_engaged, "n_cells": len(R.TRAINED_CELLS),
        "zero_engagement_disposition_doc62_SS9_4": (
            None if n_cells_engaged > 0 else
            "0/8 CELLS ENGAGED: the run produces NO RE-UPLOADING VERDICT AT ALL. "
            "The finding is about the OPTIMISER'S TREATMENT OF INTERLEAVE BLOCKS "
            "under this recipe, it is reported as such, and the compute is "
            "reported as SPENT (doc 46 SS17's precedent, honestly repeated). "
            "Every SS12 boolean is null."),
        "admissibility": adm,
        "floor_clearance": clearance,
        "verdicts": dict(binds, **dict(lifts, **init_bools)),
        "composition_cost_doc62_SS12_6": {"per_family": cc, "meta": cc_meta},
        "bug_hunt_doc62_SS15_1": {"per_cell": bug_hunt, "meta": bug_hunt_meta},
        "bounds_doc62_SS15": {
            "R1-A_R2-A_committed": 0.9625,
            "exp75_cells_grade1_bound": None,
            "note": ("doc 62 SS2.2: THERE IS NO GRADE-1 (tight, constructive, "
                     "twin-enumerating) REACHABLE-ACCURACY BOUND FOR THE EXP75 "
                     "ARMS, AND NONE IS CLAIMED. The per-referent state is "
                     "U_L(theta)|0> with U_L containing the TRAINED blocks V_j: "
                     "the encoding is data-dependent-TRAINED, the feature map is "
                     "a FAMILY INDEXED BY THE OPTIMISER'S OUTPUT, and the "
                     "committed MAX-FS-style instrument would have to be "
                     "evaluated at a point it cannot know in advance. THIS IS A "
                     "SCOPING STATEMENT MADE BEFORE THE RUN, NOT AN OMISSION "
                     "DISCOVERED AFTER IT. What replaces it is (a) a computable "
                     "RELAXATION over the degree-L span -- monotone in L, >= "
                     "0.9625, a STRICT over-estimate because unitarity carves the "
                     "achievable coefficient vectors into a non-convex proper "
                     "subset, registered COMPUTABLE-ON-DEMAND and NOT COMPUTED "
                     "(ruling 4) -- and (b) the SS8.2 closure gates, which do not "
                     "bound accuracy but CERTIFY THAT THE CIRCUIT IS THE CIRCUIT "
                     "THE ARM IS NAMED FOR."),
            "honest_note_doc62_SS15_2": (
                "doc 61's rung had a SATURATED bound and could say 'whatever we "
                "measure is a fact about TRAINING, not capacity' before spending "
                "its compute. EXP75 IS THE MIRROR IMAGE: capacity genuinely binds "
                "on the baseline (SS1.3 -- 0.9625, the programme's only committed "
                "non-saturated bound) and the intervention provably raises the "
                "class, BUT THE INSTRUMENT THAT WOULD QUANTIFY BY HOW MUCH DOES "
                "NOT APPLY. The rung trades a usable bound for a real capacity "
                "question, and both halves of that trade were stated BEFORE the "
                "run."),
        },
        "comparator": comparator,
        "no_directional_call_doc62_SS13": (
            "exp75 places NO DIRECTIONAL CALL on any criterion, stated as "
            "IGNORANCE and not as a hedge. (1) The record. (2) The universality "
            "prior is about EXPRESSIVITY, and expressivity is exactly what this "
            "programme has FOUR TIMES measured not to predict conversion (doc 56 "
            "SS18.2's ledger: frequency K, readout arity M, width, "
            "entanglement-in-the-encoding). (3) The two in-programme numbers that "
            "bear on the direction POINT OPPOSITE WAYS: the family's bound is "
            "0.9625 so supply is genuinely missing, but the arms sit 0.040 BELOW "
            "that bound so the family has not converted the supply it already "
            "has. (4) The literature registers COMPETING mechanisms, not a "
            "dominant one (Fourier locking; frequency redundancy; universality)."),
        "modal_outcome_doc62_SS12_4": (
            "THE BARS ARE HARD BY CONSTRUCTION AND THAT IS STATED RATHER THAN "
            "DISCOVERED: the baseline is already at 0.92 and the ceiling is 1.0, "
            "so `lifts` consumes 0.060 of 0.077-0.081 available headroom. It is "
            "NOT unreachable -- the wazn arms measured 0.9967 and 0.9933 on this "
            "dataset, this readout and this seed set -- but it requires a "
            "near-ceiling cell. BOTH BOOLEANS FALSE IS THE MODAL OUTCOME AND IS A "
            "RESULT, NOT A FAILURE: it reads as 'INDISTINGUISHABLE AT THIS "
            "POWER'. Combined with SS11.3's harmonic budget it becomes the "
            "programme's FIFTH AND CLEANEST supply-without-conversion instance -- "
            "cleanest because the supply was REPRESENTATIONALLY MISSING "
            "beforehand, the capacity cost was +5-10 % parameters, and the "
            "composition axis was free. THIS IS A STRONGER NEGATIVE THAN ANY OF "
            "THE FOUR LEDGER ENTRIES. It is NOT an impossibility claim; not "
            "'re-uploading does not work'; not 'small-circuit universality is "
            "false'. Two depths, unit prefactors, one map, one task family, 5 "
            "seeds -- A RECORD, NOT A THEOREM (doc 54 SS17.3)."),
        "confirm_draw_rule_doc62_SS20_2": (
            "ON ANY POSITIVE HEADLINE the confirm draw is MANDATORY and the claim "
            "MAY NOT be made on the first draw alone (doc 47 SS18; doc 58 SS20's "
            "precedent, where exactly this shape died on the second draw): "
            "generator seed + 1000, re-running the COMMITTED `exp69_datagen.py` "
            "unmodified, with ALL CELLS OF THE AFFECTED T present (both L, both "
            "inits, plus the anchor), the full V-suite and every SS8/SS9 "
            "assertion re-run, and the S-shuffle floors RE-DRAWN matched to the "
            "fresh draw. No new surface template, no new noun, no new verb. There "
            "is no warm-start donor here, so doc 58 SS16.2's donor-retraining "
            "clause does not apply."),
        "deliberately_not_registered_doc62_SS12_8": (
            "(a) any criterion reading a bound; (b) any T = 1 vs T = 2 "
            "subtraction as a criterion -- `cc` is descriptive; (c) ANY L = 2 vs "
            "L = 3 BOOLEAN, because with 5 seeds and an 0.06 bar a two-way "
            "ranking is a knife-edge magnet; (d) any advantage-shaped quantity; "
            "(e) any comparison against exp72's rescue arms or exp73's amplitude "
            "cells -- different experiments, different questions, and SS3.5 adds "
            "a specific reason for the init booleans."),
        "wall_time_sec": time.time() - t0,
    })

    if os.path.exists(a.out):
        raise RuntimeError(
            "Rule 3: %s already exists and this aggregator MAY NOT overwrite a "
            "results file of record. Move it aside deliberately if a re-run is "
            "intended." % a.out)
    json.dump(out, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp75-agg] wrote %s" % a.out)

    # ------------------------------------------------------------ the summary
    print("\n=== doc 62 SS18.4, IN ORDER ===")
    print("1. anchor_replicates_committed = %s%s"
          % (out["anchor_gate_doc62_SS12_1"]["anchor_replicates_committed"],
             "" if anchor_ok else "   <-- NOTHING ELSE MAY BE READ"))
    print("2. engagement: %d/%d cells engaged 5/5 at the %.2f rad bar"
          % (n_cells_engaged, len(R.TRAINED_CELLS), ENGAGEMENT_BAR))
    for arm in R.TRAINED_CELLS:
        e = engagement.get(arm)
        c = cells[arm]
        print("   %-24s mean=%s  engaged=%s (%s/5)  admissible=%s"
              % (arm,
                 ("%.4f" % c["mean"]) if c.get("complete") else "incomplete",
                 (e or {}).get("reupload_engaged"),
                 (e or {}).get("n_seeds_engaged"), adm[arm]["admissible"]))
    print("3. floors + binds")
    for arm in R.TRAINED_CELLS:
        b = binds["binds_%s" % arm]
        cl = clearance.get(arm)
        print("   %-24s binds=%s  floor=%s"
              % (arm, b["value"],
                 ("%.4f+-%.4f" % (cl["floor_mean"], cl["readings"]
                                  ["floor_sd_PRIMARY"]["sd"])) if cl else "MISSING"))
    print("4. THE HEADLINE")
    for T in (1, 2):
        for nm in ("reuploading_lifts_T%d" % T, "reuploading_hurts_T%d" % T):
            v = lifts.get(nm, {})
            print("   %-24s = %s   (delta=%s, bar=%s)"
                  % (nm, v.get("value"),
                     ("%+.4f" % v["delta"]) if "delta" in v else "n/a",
                     ("%.4f" % v["bar"]) if "bar" in v else "n/a"))
    print("5. init booleans")
    for nm in sorted(init_bools):
        print("   %-24s = %s" % (nm, init_bools[nm]["value"]))
    print("6. composition cost (DESCRIPTIVE, no boolean)")
    for k, v in cc.items():
        print("   cc_%-14s = %s" % (k, ("%+.4f" % v["cc"]) if v else None))
    return 0


def arbiter(comparator, T, best_cell, delta):
    """doc 62 SS10.3 -- THE ARBITER RULE, and SS12.4.1 makes it BINDING on any
    TRUE `lifts`. Tier (ii) is the SPAN-MATCHED classical model at degree L; tier
    (iii) is the COMMITTED 9-dim single-encode comparator, value 0.9200, IMPORTED
    AND NEVER RE-FITTED."""
    tier3 = 0.9200
    row = None
    tier2 = None
    if comparator:
        try:
            L = R.ARM_CFG[best_cell]["L"]
            tier2 = float(comparator["tiers"]["span_matched"]["L%d" % L]
                          ["T%d" % T]["mean"])
        except (KeyError, TypeError, ValueError):
            tier2 = None
    gap = None if tier2 is None else tier2 - tier3
    if tier2 is not None:
        if delta >= DELTA_BAR and gap >= DELTA_BAR:
            row = ("OK -- THE ADDED FREQUENCIES CARRY INFORMATION AND THE CIRCUIT "
                   "CONVERTED IT. This is the ONLY configuration in which a lift "
                   "may be reported as a re-uploading win.")
        elif delta >= DELTA_BAR:
            row = ("WARNING -- THE CIRCUIT'S GAP EXCEEDS THE INFORMATION GAP: the "
                   "lift is a BASELINE-TRAINABILITY ARTEFACT (the committed "
                   "single-encode arm under-trained its own class; doc 62 SS1.4 "
                   "measured 0.04 of unconverted supply) and `reuploading_lifts` "
                   "MAY NOT BE CLAIMED AS A RE-UPLOADING RESULT. Reported with "
                   "both numbers.")
        elif gap >= DELTA_BAR:
            row = ("WARNING -- SUPPLY IS REAL AND EXTRACTABLE CLASSICALLY; THE "
                   "CIRCUIT DID NOT CONVERT IT. The trainability reading, doc 55 "
                   "SS8.2's middle row, and the programme's FIFTH instance.")
        else:
            row = ("THE EXTRA FREQUENCIES CARRY NO USABLE INFORMATION FOR THIS "
                   "TASK AT THIS SAMPLE SIZE, and no circuit claim is available "
                   "either way.")
    return {
        "tier_iii_committed_9dim": tier3,
        "tier_ii_span_matched_at_degree_L": tier2,
        "information_gap_tier_ii_minus_tier_iii": gap,
        "circuit_delta": delta, "bar": DELTA_BAR,
        "row": row,
        "comparator_missing": bool(tier2 is None),
        "restriction_qualifier_doc62_SS10_4": (
            "BOTH HALVES TRAVEL WITH EVERY TIER (ii) NUMBER, AND HERE THEY POINT "
            "IN OPPOSITE DIRECTIONS. (a) SUPERSET IN COEFFICIENT SPACE: tier (ii) "
            "ranges over ALL coefficient vectors in the degree-L span, while "
            "unitarity restricts the circuit to a NON-CONVEX PROPER SUBSET, so "
            "the comparator is an UPPER reference and a circuit below it is not "
            "evidence the class is insufficient. (b) RESTRICTION ACROSS "
            "CONDITIONING GROUPS: the circuit carries one form per conditioning "
            "group (12 at T = 1, 72 at T = 2) while the comparator carries one "
            "form total, so the comparator is simultaneously a WEAKER model in "
            "that dimension. THE HONEST SUMMARY IS THEREFORE 'TIE IS THE "
            "STRONGEST AVAILABLE STATEMENT', NOT 'THE CIRCUIT BEAT THE CLASS'."),
        "not_vacuous_doc62_SS10_2": (
            "the joint dimensions are 25 (L = 2) and 49 (L = 3) against 660 "
            "train+val items -- a ratio of 13:1 at worst -- so NO REGULARISED "
            "TIER IS BOUGHT, the fit is the committed deterministic "
            "unregularised logistic regression at the committed fit seed, and doc "
            "61 SS8.3's regularisation qualifier DOES NOT APPLY to this rung."),
        "registered_prediction_doc62_SS10_5": (
            "THE COMPARATORS MATCH OR EXCEED EVERY ARM (doc 47 SS14; doc 52 "
            "SS1.1; measured on every arm of exp67, exp68, exp69 and exp70). "
            "Recorded so a tie is NOT reported as a surprise."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
