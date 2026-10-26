# -*- coding: utf-8 -*-
"""exp76_aggregate.py -- doc 63 SS12 / SS18.4. THE ONLY WRITER OF
`results_exp76.json`.

A COLD, SEPARATE PASS (SS20.4). It shares no state with the build session: it
reads the 120 per-run checkpoints, the anchor gate, the smoke report, the
comparator and the certificate, and it writes one results file. Rule 3 forbids
overwriting any existing results file.

    1. wait:   ls exp76_ckpt/*.json | grep -v comparator | wc -l   == 120
               and all five worker logs end in "[exp76 wN] done"
    2. run:    exp76_run.py --anchor-report
    3. run:    exp76_comparator.py
    4. run:    exp76_certificate.py
    5. run:    exp76_aggregate.py           <- this file
    6. read:   results_exp76_anchors.json first, then results_exp76.json

SS18.4's GATING ORDER IS ENFORCED MECHANICALLY, so no verdict is read past a
missing floor:

    anchor -> inert-padding control -> comparator_flat_in_n -> floor exists
    -> ONLY THEN binds -> gap_stable/degrades (with SS10.5's arbiter)
    -> composition_free_at_scale -> inventory_costs -> grade 3
    -> SS11's descriptives -> SS9's certificate (read LAST, gating nothing).

A CELL WHOSE FLOOR HAS NOT BEEN RUN HAS NO VERDICT AND ITS BOOLEANS ARE STORED
AS `null`, NEVER AS FALSE. The same disposition applies wherever the doc
registers it: SS12.3 (a failed `comparator_flat_in_n` makes SS12.5's booleans
null), SS12.9.4 (an unreadable grade 3 is null), SS10.4's vacuous scope.

TWO THINGS THE REGISTRATION SETTLES AND THIS FILE IMPLEMENTS AS SETTLED:

  * SS12.4.1 -- the 0.65 `binds` bar sits ABOVE faail_camel's 0.6426 pair-level
    grade-1 bound, so `binds` is registered as PLAUSIBLY UNREACHABLE BY
    ARITHMETIC across the 519-noun cells. It is retained unchanged (doc 55
    SS6.2's provenance rule forbids re-choosing a bar) and NO HEADLINE CRITERION
    IS GATED ON IT. `binds_band32_T2 = FALSE` and `binds_n2_T2 = FALSE` are the
    registered EXPECTATIONS and are not failures of anything.

  * SS12.9.3 -- grade 3 is READ IFF THE REFERENCE ARM CLEARS ITS OWN
    PROCEDURE-MATCHED FLOOR, not iff it binds. SS18.4's gating list still carries
    the older parenthetical "(only after binds_n2_T2 ... is TRUE)"; SS12.4.1 and
    SS12.9.3 both state in terms that the gate WAS REWRITTEN to floor clearance
    precisely because gating on `binds` would make grade 3 unreadable BY
    ARITHMETIC on the one run in this programme's history that finally has a
    target for it. THIS FILE IMPLEMENTS SS12.9.3 AND FLAGS SS18.4's stale
    parenthetical rather than inheriting it.

NO exp76 NUMBER MAY APPEAR IN A SENTENCE CONTAINING "ADVANTAGE" (SS0.3, SS16).
NO exp76 NUMBER MAY BE QUOTED AS EVIDENCE THAT AN ENCODING "CARRIES WAZN"
(SS16, SS3.6.3, SS21 item 9).
"""

import argparse
import glob
import json
import os
import platform
import sys

import numpy as np
import torch

import exp76_run as R
import exp76_datagen as D76

OUT = "results_exp76.json"
ANCHORS = R.ANCHOR_OUT
SMOKE = R.SMOKE_OUT
CERT = "diagnostic_exp76_certificate.json"
COMPARATOR = os.path.join(R.CKPT_DIR, "comparator.json")

BINDS_BAR = R.BINDS_BAR            # 0.65, doc 54 SS7, INHERITED
DELTA_BAR = R.DELTA_BAR            # 0.06 = 2 x binomial SD at n_test 300
FLOOR_MULT = R.FLOOR_MULTIPLE      # 2 SD, floor's own SD PRIMARY
BUG_HUNT_BOUND = 0.9625            # SS12.8, the committed R-A bound
BUG_HUNT_MARGIN = 0.06


def agg(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": [float(x) for x in a], "n_seeds": len(a)}


def load_runs(ckpt_dir, seeds):
    runs = {}
    for arm in R.ALL_ARMS:
        per = []
        for s in range(seeds):
            f = os.path.join(ckpt_dir, "%s_s%d.json" % (arm, s))
            if not os.path.exists(f):
                raise RuntimeError(
                    "run incomplete: %s is missing. doc 63 SS20.4 step 1 -- "
                    "aggregation runs only when all 120 checkpoints exist." % f)
            per.append(json.load(open(f, encoding="utf-8")))
        runs[arm] = per
    return runs


def three_sd_readings(arm_vals, floor_vals):
    """doc 63 SS12: floor multiple 2 SD, THE FLOOR'S OWN SD IS PRIMARY, ALL
    THREE readings computed and reported, and a cell whose readings DISAGREE is
    reported as disagreeing rather than resolved silently (doc 55 SS6.6; doc 58
    SS9.3; doc 60's ruling 5, PRIMARY-DECIDES)."""
    a, f = np.asarray(arm_vals, float), np.asarray(floor_vals, float)
    am, fm_ = float(a.mean()), float(f.mean())
    asd = float(a.std(ddof=1))
    fsd = float(f.std(ddof=1))
    psd = float(np.sqrt((asd ** 2 + fsd ** 2) / 2.0))
    out = {}
    for name, sd in (("floor_sd_PRIMARY", fsd), ("arm_sd", asd),
                     ("pooled_sd", psd)):
        out[name] = {"sd": sd, "threshold": fm_ + FLOOR_MULT * sd,
                     "clears": bool(am >= fm_ + FLOOR_MULT * sd)}
    verdicts = {k: v["clears"] for k, v in out.items()}
    out["readings_disagree"] = bool(len(set(verdicts.values())) > 1)
    out["clears_PRIMARY"] = out["floor_sd_PRIMARY"]["clears"]
    out["arm_mean"], out["floor_mean"] = am, fm_
    # SS18.5's contamination branch
    out["floor_contaminated"] = bool(fm_ >= am - 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise RuntimeError("Rule 3: %s already exists and no results file may "
                           "be overwritten" % a.out)
    n_ck = len([p for p in glob.glob(os.path.join(a.ckpt_dir, "*.json"))
                if "comparator" not in os.path.basename(p)])
    runs = load_runs(a.ckpt_dir, a.seeds)
    smoke = json.load(open(SMOKE, encoding="utf-8"))
    anch = json.load(open(ANCHORS, encoding="utf-8"))
    comp = json.load(open(COMPARATOR, encoding="utf-8"))
    cert = json.load(open(CERT, encoding="utf-8"))

    rep = {
        "experiment": "exp76 -- scaling the composition-free family (doc 63)",
        "spec": R.SPEC,
        "question": ("The single-angle family binds a category to a role at 2 "
                     "referents, 2 events and 24 nouns, and composition costs "
                     "it 0.004. Does any of that survive when the cast grows, "
                     "the story lengthens, and the noun inventory becomes a "
                     "real one -- and if it degrades, is the degradation in the "
                     "TASK or in the CIRCUIT?"),
        "aggregated_from": os.path.join(a.ckpt_dir, "*.json"),
        "aggregator": "exp76_aggregate.py",
        "n_checkpoints": n_ck,
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "sd_convention": "sample sd, ddof=1",
        "smoke_results": SMOKE, "anchor_results": ANCHORS,
        "vsuite_results": D76.VALIDATION_OUT,
        "comparator_results": COMPARATOR, "certificate_results": CERT,
        "training": {"lr": R.LR, "batch": R.BATCH, "max_epochs": R.MAX_EPOCHS,
                     "optimiser": "Adam", "rng_base": R.RNG_BASE,
                     "model_selection": "best in-distribution val accuracy",
                     "seeds_vary": ("OPTIMISER INITIALISATION ONLY -- the "
                                    "features are deterministic constants"),
                     "init": "the committed uniform init ONLY (doc 63 SS7.4)",
                     "n_params": R.N_PARAMS,
                     "n_params_note": ("120 at n = 2 AND at n = 7. exp76 is the "
                                       "first rung in this programme where the "
                                       "intervention adds ZERO parameters.")},
        "n_runs": len(R.ALL_ARMS) * a.seeds,
        "arms": {}, "floors": {}, "cells": {},
    }

    # ------------------------------------------------- the raw per-arm tables
    for arm, per in runs.items():
        cfg = R.ARM_CFG[R.base_arm(arm)]
        t = agg([r["heldout_test_acc"] for r in per])
        t.update({
            "kind": cfg["kind"], "cell": cfg["cell"], "axis": cfg["axis"],
            "n": cfg["n"], "T": cfg["T"],
            "n_active": min(cfg["n"], cfg["T"] + 1),
            "inventory": cfg["inventory"], "category": cfg["category"],
            "role": cfg["role"], "n_params": per[0]["n_params"],
            "wires": per[0]["wires"],
            "val_per_seed": [r["val_acc"] for r in per],
            "train_per_seed": [r["train_acc"] for r in per],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in per],
            "val_minus_heldout_per_seed": [r["val_minus_heldout"] for r in per],
            "mean_val_minus_heldout": float(np.mean(
                [r["val_minus_heldout"] for r in per])),
            "gate_met_per_seed": [r["gate_met"] for r in per],
            "dataset": per[0]["dataset"],
            "dataset_sha256": per[0]["dataset_sha256"],
            "ckpt_sha256": [r["ckpt_sha256"] for r in per],
            "wall_sec_total": float(sum(r["wall_sec"] for r in per))})
        (rep["floors"] if arm.startswith("S-shuffle") else rep["arms"])[arm] = t

    # ============================== SS18.4 STEP 1 -- THE ANCHOR, BLOCKING =====
    anchor_ok = bool(anch["anchor_replicates_exp69"])
    # ============================== SS18.4 STEP 2 -- THE PADDING CONTROL ======
    padding_ok = bool(smoke["inert_padding_doc63_SS8_3"]["holds"])
    # ============================== SS18.4 STEP 3 -- THE COMPARATOR ===========
    flat_ok = bool(comp["comparator_flat_in_n"])

    blocking = {"anchor_replicates_exp69": anchor_ok,
                "inert_padding_control_holds": padding_ok,
                "comparator_flat_in_n": flat_ok}
    run_readable = anchor_ok and padding_ok
    n_axis_readable = run_readable and flat_ok

    def nullable(value, ok, why):
        """SS18.4 / SS12.3 / SS12.9.4: an unreadable boolean is stored as null,
        NEVER as FALSE."""
        return (value if ok else None), (None if ok else why)

    # ============================== SS18.4 STEP 4-5 -- FLOORS, THEN binds =====
    for cell in R.TRAINED_CELLS:
        cfg = D76.CELL_CFG[cell]
        arm_v = rep["arms"][cell]["per_seed"]
        fl_v = rep["floors"][R.FLOOR_OF[cell]]["per_seed"]
        sdr = three_sd_readings(arm_v, fl_v)
        mean = rep["arms"][cell]["mean"]
        clears = sdr["clears_PRIMARY"]
        b = bool(mean >= BINDS_BAR) and clears
        rep["cells"][cell] = {
            "n": cfg["n"], "T": cfg["T"], "inventory": cfg["inv"],
            "category": cfg["cat"], "axis": cfg["axis"],
            "heldout_mean": mean, "heldout_sd": rep["arms"][cell]["sd"],
            "floor_mean": sdr["floor_mean"],
            "floor_sd": sdr["floor_sd_PRIMARY"]["sd"],
            "comparator_tier_i": comp["arms"][cell]["i_raw"]["mean"],
            "comparator_tier_ii": comp["arms"][cell]["ii_committed"]["mean"],
            "comparator_tier_iii": comp["arms"][cell]["iii_distractor"]["mean"],
            "gap_tier_ii": mean - comp["arms"][cell]["ii_committed"]["mean"],
            "binds": (b if run_readable else None),
            "binds_part_absolute_bar": bool(mean >= BINDS_BAR),
            "binds_part_floor_clearance": clears,
            "floor_readings": sdr,
            "floor_contaminated": sdr["floor_contaminated"],
            "certificate": cert["per_cell"].get(cell),
        }
        if sdr["floor_contaminated"]:
            rep["cells"][cell]["contamination_note"] = (
                "doc 63 SS18.5 / SS14: the procedure is fitting the SPLIT, not "
                "the labels. THE CELL IS REPORTED AS FLOOR-CONTAMINATED WITH "
                "BOTH NUMBERS AND IS NOT READ (doc 58 SS10; doc 61 SS15.5).")

    # ============================== SS18.4 STEP 6 -- SS12.5's HEADLINE ========
    gap = {}
    for cell in R.TRAINED_CELLS:
        cfg = D76.CELL_CFG[cell]
        if cfg["axis"] != "grid":
            continue
        gap[(cfg["n"], cfg["T"])] = rep["cells"][cell]["gap_tier_ii"]
    verdicts, reasons = {}, {}
    for T in (2, 4, 6):
        ns = sorted(n for (n, t) in gap if t == T)
        gs = [gap[(n, T)] for n in ns]
        spread = float(max(gs) - min(gs))
        endpoint = float(gap[(ns[-1], T)] - gap[(ns[0], T)])
        why = ("doc 63 SS12.3: `comparator_flat_in_n` FAILED, the n-axis cells "
               "are NOT comparable, and SS12.5's booleans are stored as null, "
               "NEVER as FALSE. The finding is a GENERATOR DEFECT (SS14).")
        v, r = nullable(bool(spread <= DELTA_BAR), n_axis_readable, why)
        verdicts["gap_stable_in_n_T%d" % T] = v
        reasons["gap_stable_in_n_T%d" % T] = r or (
            "spread(T=%d) = %.4f over n = %r against the 0.06 bar "
            "(2 x binomial SD at n_test = 300)" % (T, spread, ns))
        v2, r2 = nullable(bool(endpoint <= -DELTA_BAR), n_axis_readable, why)
        verdicts["gap_degrades_in_n_T%d" % T] = v2
        reasons["gap_degrades_in_n_T%d" % T] = r2 or (
            "gap(n=%d) - gap(n=%d) = %.4f against the -0.06 bar"
            % (ns[-1], ns[0], endpoint))
        rep.setdefault("gap_by_T", {})["T%d" % T] = {
            "n_values": ns, "gap": gs, "spread": spread,
            "endpoint_delta": endpoint,
            "circuit_heldout": [rep["cells"]["n%d_T%d" % (n, T)]["heldout_mean"]
                                for n in ns],
            "comparator_tier_ii": [
                rep["cells"]["n%d_T%d" % (n, T)]["comparator_tier_ii"]
                for n in ns]}
        if T == 2:
            rep["gap_by_T"]["T2"]["thinness_note_doc63_SS12_5_1"] = (
                "THE T = 2 PAIR IS THIN AND IS REGISTERED AS THIN: n <= T + 1 "
                "allows only n in {2, 3}, one step of cast size. "
                "`gap_stable_in_n_T2` is a TWO-POINT statement and is the "
                "weakest of the three; T = 4 and T = 6 are where the n axis is "
                "actually measured, and any summary that quotes the three "
                "booleans as equivalent evidence is WRONG.")

    # ============================== SS18.4 STEP 7 -- SS12.6 ===================
    cc = {}
    for n in sorted({D76.CELL_CFG[c]["n"] for c in R.TRAINED_CELLS
                     if D76.CELL_CFG[c]["axis"] == "grid"}):
        ms = [rep["cells"][c]["heldout_mean"] for c in R.TRAINED_CELLS
              if D76.CELL_CFG[c]["axis"] == "grid" and D76.CELL_CFG[c]["n"] == n]
        cc[n] = {"n_T_points": len(ms), "heldout_means": ms,
                 "cc": (float(max(ms) - min(ms)) if len(ms) > 1 else None)}
    cfree = (bool(cc[2]["cc"] <= DELTA_BAR) and bool(cc[4]["cc"] <= DELTA_BAR))
    v, r = nullable(cfree, run_readable,
                    "the run stopped at SS18.4's blocking gates")
    verdicts["composition_free_at_scale"] = v
    reasons["composition_free_at_scale"] = r or (
        "cc(2) = %.4f over T in {2,4,6} and cc(4) = %.4f over T in {4,6}, both "
        "against the 0.06 bar. THE CRITERION READS ONLY n = 2 AND n = 4 and the "
        "reason is the grid's SHAPE: n <= T + 1 makes the grid TRIANGULAR, so "
        "the frontier cells (3,2), (5,4) and (7,6) each exist at exactly one T "
        "and no T-axis spread is defined for them (SS12.6, SS23 item 4). The "
        "composition question is therefore answered at cast sizes 2 and 4, NOT "
        "at 7 -- and 7 is exactly where branch B2 predicts the collapse."
        % (cc[2]["cc"], cc[4]["cc"]))
    rep["composition_cost_doc63_SS12_6"] = {
        "cc_by_n": cc, "committed_cc_A_at_n2_T1_T2": R.COMMITTED_CC_A,
        "ladder_doc63_SS0_2": {
            "single_angle_FA": 0.0040, "CZ_across": 0.032, "eps_register": 0.132,
            "old_map_register": 0.253, "A3_amplitude_identity_init": 0.2627,
            "entangled_register": 0.310, "A3_amplitude_uniform_init": 0.3447},
        "branches_doc63_SS12_6": {
            "B1_freedom_is_the_encodings": ("cc(n) <= 0.06 at every n; the "
                                            "coefficient budget does not grow "
                                            "with T, only depth does"),
            "B2_freedom_was_the_settings": ("at n = 2 both events act on the "
                                            "SAME two wires, so exp69's "
                                            "'composition' was two blocks on "
                                            "one register; at n > 2 the queried "
                                            "pair is entangled with n_active-2 "
                                            "other wires the readout must trace "
                                            "out, which is a genuinely "
                                            "different operation")},
        "no_directional_call_doc63_SS13": True}

    # ============================== SS18.4 STEP 8 -- SS12.7 ===================
    inv = {}
    for i24, i519 in (("I24_n2_T2", "n2_T2"), ("I24_n4_T4", "n4_T4")):
        d = rep["cells"][i24]["heldout_mean"] - rep["cells"][i519]["heldout_mean"]
        inv[i519] = {"I24_mean": rep["cells"][i24]["heldout_mean"],
                     "I519_mean": rep["cells"][i519]["heldout_mean"],
                     "delta_I24_minus_I519": d,
                     "costs_at_this_cell": bool(d >= DELTA_BAR),
                     "grade1_bound_I519": smoke["committed_imports_doc63_SS8_6"]
                                               ["grade1_bounds_doc63_SS2_6"]
                                               ["faail_camel"],
                     "grade1_bound_I24": None,
                     "FLAGGED_no_I24_bound_exists": (
                         "AMBIGUITY F7. SS6.3 states that 'both levels' grade-1 "
                         "bounds are computed in the pre-work (SS2.6) so the "
                         "capacity change is known before the accuracy change is "
                         "seen', but `diagnostic_scaling_bounds.json` (produced "
                         "2026-08-29) carries NO I24 contrast row -- the 24-noun "
                         "subset is drawn at build time at the registered seed "
                         "SEED + 24 and did not exist when the instrument ran. "
                         "NO I24 BOUND IS INVENTED HERE. This costs the run "
                         "nothing mechanical: SS12's preamble makes it "
                         "mechanical that NO CRITERION READS A BOUND, and SS15 "
                         "lists the bounds as CONTEXT ONLY. It costs the "
                         "READING the separation SS14 wants between 'the "
                         "inventory made the TASK harder' and 'the inventory "
                         "made the TRAINING harder', so `inventory_costs` is "
                         "reported WITHOUT that discriminator and the gap is "
                         "flagged rather than papered over. Computing it is one "
                         "bound plus a reused null (~15 s) and is a decision "
                         "for the orchestrator, not for this aggregator.")}
    icosts = any(v["costs_at_this_cell"] for v in inv.values())
    v, r = nullable(bool(icosts), run_readable,
                    "the run stopped at SS18.4's blocking gates")
    verdicts["inventory_costs"] = v
    reasons["inventory_costs"] = r or (
        "SS12.7: `inventory_costs = (heldout_mean(I24) - heldout_mean(I519) >= "
        "0.06) at (2,2) OR at (4,4), reported per cell`. Measured deltas: %s. "
        "AMBIGUITY FLAGGED: SS12.7 writes 'at (2,2) or at (4,4)' without "
        "settling whether the boolean is the OR or a per-cell pair; this file "
        "implements the OR and reports BOTH cells so either reading is "
        "recoverable. Read together with the two levels' pre-work bounds (SS2.6) "
        "so a drop that TRACKS the bound is separated from one that does not."
        % {k: round(v2["delta_I24_minus_I519"], 4) for k, v2 in inv.items()})
    rep["inventory_axis_doc63_SS12_7"] = inv

    # ============================== SS18.4 STEP 9 -- GRADE 3 =================
    ref, tgt = "n2_T2", "band32_n2_T2"
    g1 = smoke["committed_imports_doc63_SS8_6"]["grade1_bounds_doc63_SS2_6"]
    ref_clears = rep["cells"][ref]["binds_part_floor_clearance"]

    def headroom(cell, bound):
        return (rep["cells"][cell]["heldout_mean"] - 0.5) / (bound - 0.5)

    hc_ref = headroom(ref, g1["faail_camel"])
    hc_tgt = headroom(tgt, g1["band_32arc"])
    readable = run_readable and ref_clears
    v, r = nullable(bool(hc_tgt < hc_ref), readable,
                    "doc 63 SS12.9.4: the reference arm does NOT clear its own "
                    "procedure-matched floor, so BOTH NUMERATORS ARE NOISE and "
                    "the ratio has no content. Stored as null, NEVER as FALSE, "
                    "and recorded as a FOURTH distinct reason grade 3 went "
                    "untested (doc 55 SS19.2, doc 56 SS9.3, doc 61 SS2.5 -- "
                    "'testable at design time, unreadable at read-out').")
    verdicts["grade3_under_conversion_flag"] = v
    reasons["grade3_under_conversion_flag"] = r or (
        "headroom_conversion(band_32arc @ 2,2) = %.4f against "
        "headroom_conversion(faail_camel @ 2,2) = %.4f, with grade-1 bounds "
        "0.5298 and 0.6426 FROZEN IN SS2.6 BEFORE ANY ARM RAN. The flag is a "
        "DIRECTIONAL statement at its registered grade, not a quantitative "
        "prediction." % (hc_tgt, hc_ref))
    rep["grade3_doc63_SS12_9"] = {
        "reference_cell": ref, "target_cell": tgt,
        "grade1_bounds_frozen_in_SS2_6": g1,
        "headroom_conversion": {ref: hc_ref, tgt: hc_tgt},
        "read_condition_doc63_SS12_9_3": {
            "rule": ("grade 3 is READ iff the reference arm clears its own "
                     "procedure-matched floor: heldout_mean(faail_camel @ 2,2) "
                     ">= floor_mean + 2 * floor_sd"),
            "reference_clears_its_floor": ref_clears, "readable": readable,
            "why_not_binds": (
                "SS12.4.1 makes a `binds` gate SELF-DEFEATING: the 0.65 bar "
                "sits ABOVE faail_camel's 0.6426 pair-level bound, so gating on "
                "it would make grade 3 unreadable BY ARITHMETIC on the one run "
                "in this programme's history that finally has a target for it. "
                "Floor clearance answers 'did this arm learn anything from the "
                "labels', which is the only precondition a conversion ratio "
                "needs. This is NOT a weakened bar; it is the CORRECT bar for "
                "this question, and 0.65 is not re-chosen -- it is not used."),
            "FLAGGED_stale_text_in_SS18_4": (
                "SS18.4's gating list still reads 'grade 3 (only after "
                "binds_n2_T2 for faail_camel is TRUE, SS12.9.3)'. That "
                "parenthetical is superseded by SS12.9.3's own text and by "
                "SS12.4.1's ruling that the gate WAS REWRITTEN to floor "
                "clearance. This aggregator implements SS12.9.3 and flags the "
                "stale wording rather than inheriting it.")},
        "why_the_headroom_form_doc63_SS12_9_1": (
            "the obvious form heldout/bound is DEGENERATE NEAR CHANCE -- both "
            "terms compress toward 0.5, so an arm at 0.52 against a 0.5298 "
            "bound scores 0.98 while an arm at 0.60 against a 0.6426 bound "
            "scores 0.93, and the flag would fire BACKWARDS for a purely "
            "arithmetic reason. The headroom form divides the excess over "
            "chance by the AVAILABLE excess over chance."),
        "why_both_cells_are_at_n2_doc63_SS12_9_2": (
            "SS2.8: NO grade-1 bound exists at n > 2 -- `capacity_check.py`'s "
            "scope statement is that the instrument is not a bound on tasks "
            "with more than two referents. A conversion criterion needs a bound "
            "in its denominator, so grade 3 can only be read where the "
            "pre-work's instrument applies."),
        "registered_expectations_doc63_SS12_9_4": {
            "binds_band32_T2": False, "binds_n2_T2": False,
            "why": ("band_32arc's bound is 0.5298, so an arm binding at 0.65 "
                    "would exceed its own reachable ceiling by 0.12 and would "
                    "be a bug_hunt event; faail_camel's is 0.6426 and the 0.65 "
                    "bar meets a 519-noun inventory. NEITHER IS A FAILURE OF "
                    "ANYTHING.")},
        "scope": ("a PREDICTION-GRADE result about the BOUND instrument -- NOT "
                  "a finding about wazn, about morphology, or about scaling. It "
                  "says the bound's null percentile carries information about "
                  "trainability; nothing more. One target, one cell, five "
                  "seeds, after three rungs with none: real, and thin "
                  "(SS23 item 3a).")}

    # ============================== SS12.8, INERT ============================
    over = {c: rep["cells"][c]["heldout_mean"] - BUG_HUNT_BOUND
            for c in R.TRAINED_CELLS}
    verdicts["bug_hunt_triggered"] = bool(
        any(v2 > BUG_HUNT_MARGIN for v2 in over.values()))
    reasons["bug_hunt_triggered"] = (
        "SS12.8: computed and stored for completeness and REGISTERED AS "
        "STRUCTURALLY INERT. SS2.8 establishes that NO grade-1 bound exists at "
        "n > 2, so there is nothing to exceed; at n = 2 it reads against 0.9625 "
        "and fires above 1.0225, which is unreachable. NO READING DEPENDS ON IT "
        "(doc 58 SS9.5; doc 61 SS12.1; doc 62 SS15.1 -- fourth use).")
    rep["bug_hunt_doc63_SS12_8"] = {
        "bound_read_against": BUG_HUNT_BOUND, "margin": BUG_HUNT_MARGIN,
        "fires_above": BUG_HUNT_BOUND + BUG_HUNT_MARGIN,
        "excess_by_cell": over, "structurally_inert": True}

    # ============================== SS18.4 STEP 11 -- THE CERTIFICATE, LAST ===
    verdicts["entanglement_area_law_like"] = \
        cert["verdicts"]["entanglement_area_law_like"]
    reasons["entanglement_area_law_like"] = \
        cert["verdicts"]["entanglement_area_law_like_reason"]
    verdicts["mps_feasible_at_chi64"] = cert["verdicts"]["mps_feasible_at_chi64"]
    reasons["mps_feasible_at_chi64"] = \
        cert["verdicts"]["mps_feasible_at_chi64_is_INERT"]

    # ------------------------------------------------------- binds, 11 of them
    for cell in R.TRAINED_CELLS:
        verdicts["binds_%s" % cell] = rep["cells"][cell]["binds"]
        reasons["binds_%s" % cell] = (
            "conjunctive: heldout_mean %.4f >= 0.65 is %s AND >= floor_mean + "
            "2 * floor_sd (%.4f) is %s. SS12.4.1: the 0.65 bar sits ABOVE "
            "faail_camel's 0.6426 grade-1 bound, so a FALSE here is registered "
            "as plausibly unreachable BY ARITHMETIC and is NOT the headline; no "
            "headline criterion is gated on `binds`."
            % (rep["cells"][cell]["heldout_mean"],
               rep["cells"][cell]["binds_part_absolute_bar"],
               rep["cells"][cell]["floor_readings"]["floor_sd_PRIMARY"]
                                 ["threshold"],
               rep["cells"][cell]["binds_part_floor_clearance"]))

    verdicts.update(blocking)
    reasons["anchor_replicates_exp69"] = (
        "doc 63 SS6.4, the SIXTH enforced use; bit_exact_all = %s. IF THIS "
        "FAILS the exp76 pipeline has drifted, every scaling number is a "
        "comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS "
        "READ." % anch["bit_exact_all"])
    reasons["inert_padding_control_holds"] = (
        "doc 63 SS8.3 / SS12.2, asserted at smoke with the all-wires-readout "
        "build that MUST fail it: max |padded - unpadded| = %.3e against the "
        "1e-6 bar, and the broken build moved by %.3e. If it fails, SS4.4's "
        "theorem is not implemented and every n_active label in the run is wrong."
        % (smoke["inert_padding_doc63_SS8_3"]["max_abs_diff"],
           smoke["inert_padding_doc63_SS8_3"]
                ["failing_direction_all_wires_readout"]["max_abs_diff"]))
    reasons["comparator_flat_in_n"] = comp["flat_reading"]

    rep["verdicts"] = verdicts
    rep["verdict_reasons"] = reasons
    rep["gating_order_doc63_SS18_4"] = [
        "anchor", "inert-padding control", "comparator_flat_in_n",
        "floor exists", "binds", "gap_stable/degrades (SS10.5's arbiter)",
        "composition_free_at_scale", "inventory_costs", "grade 3",
        "SS11 descriptives", "SS9 certificate (read LAST, gating nothing)"]
    rep["null_not_false_disposition"] = (
        "a cell whose floor has not been run has NO verdict and its booleans "
        "are stored as null, NEVER as FALSE (SS18.4). The same disposition "
        "applies to SS12.3 (a failed comparator_flat_in_n makes SS12.5's "
        "booleans null), SS12.9.4 (an unreadable grade 3), and SS10.4's vacuous "
        "single-n groups.")
    rep["arbiter_doc63_SS10_5"] = {
        "gap_within_0_06_comparator_flat": (
            "THE FAMILY SCALES. The circuit tracks the extractable information "
            "at every cast size and composition-freedom is not an artefact of "
            "n = 2."),
        "gap_degrades_comparator_flat": (
            "CIRCUIT-SIDE, AND CLEANLY SO. The information is unchanged and "
            "linearly extractable at every n (SS4.5.1, SS10.4); the circuit "
            "stops converting it. This is the programme's cleanest trainability "
            "instance BECAUSE THE TASK DID NOT CHANGE."),
        "gap_degrades_and_comparator_degrades": (
            "`comparator_flat_in_n` has FAILED and the cells are not "
            "comparable. NOT A CIRCUIT FINDING OF ANY KIND; hunt the generator."),
        "circuit_exceeds_comparator": (
            "reported with BOTH halves of SS10.6's qualifier. It has not "
            "happened on any arm of exp67, exp68, exp69, exp70 or exp73.")}
    rep["directional_call_doc63_SS13"] = {
        "call_placed": None,
        "record": "1 held (twice-drawn), 2 missed",
        "why_none": ("SS13.2: the record does not support a posture; the two "
                     "committed cc values point in OPPOSITE directions; the n "
                     "axis has no in-programme precedent at all; and SS4.4's "
                     "theorem cuts both ways -- inert wires do nothing (stability) "
                     "while the active ones are ALL entangled into the readout's "
                     "traced-out complement (degradation).")}
    rep["scope_doc63_SS16"] = [
        "NO QUANTUM READING OF ANY KIND. No exp76 number may appear in a "
        "sentence containing 'advantage'.",
        "Not a statement about scaling in general: n_active in {2,3,4,5,7}, T in "
        "{2,4,6}, one qubit per referent, one category, one surface, two "
        "inventory sizes, 5 seeds, dense simulation.",
        "Not a statement about barren plateaus: no gradient-variance-versus-"
        "qubit-count measurement is made, and the parameter count does not grow "
        "with qubit count, so the standard construction does not even apply.",
        "Not an MPS result: SS9 emits a certificate about THESE states.",
        "Not novel-filler generalisation: the split is over unseen "
        "pairings/casts, not unseen nouns, and V2 is reported UN-GATED for that "
        "reason.",
        "NOT A WAZN FINDING. The memberships are PROGRAMMATIC WITH SPOT-CHECK "
        "and a different provenance class from doc 51 SS7's user-confirmed "
        "contrast. SS3.6.2's bands establish that the classes' separability is "
        "NOT an artefact of arc count; SS3.6.3 establishes that the bands do NOT "
        "distinguish morphological structure from SEMANTIC COHERENCE, and "
        "nisba_yaa's 0.8002 is the standing reason to think the semantic channel "
        "is live. NO exp76 NUMBER MAY BE QUOTED AS EVIDENCE THAT AN ENCODING "
        "'CARRIES WAZN'. SS21 item 9's control stays registered-open.",
        "Not a discourse or anaphora result: Q76 resolves the query by VERB "
        "IDENTITY (SS4.2 C2).",
        "SS9.6 item 3: `gap_stable_in_n` TRUE together with "
        "`entanglement_area_law_like` TRUE is ONE FINDING WEARING TWO NAMES and "
        "the two may NOT be presented as mutual support.",
    ]
    rep["confirm_draw_obligation_doc63_SS20_2"] = (
        "ON ANY POSITIVE HEADLINE: a confirm draw at generator seed + 1000, "
        "re-running the COMMITTED exp76_datagen.py unmodified, with ALL cells of "
        "the affected axis present, the full V-suite and every SS8 and SS10.4 "
        "assertion re-run, and the S-shuffle floors RE-DRAWN matched to the "
        "fresh draw. doc 47 SS18 is standing and doc 58 SS20 is the reason it is "
        "not negotiable.")
    rep["deliberately_not_registered_doc63_SS12_12"] = [
        "any criterion reading a bound (SS2.8, SS15)",
        "any raw accuracy-versus-n boolean (SS10.1 is the whole reason)",
        "any (n,T) cell-versus-cell boolean beyond SS12.5's endpoints -- with 5 "
        "seeds and an 0.06 bar an 8-way ranking is a knife-edge magnet",
        "any advantage-shaped quantity (SS16)",
        "any comparison against exp70's, exp72's, exp73's or exp75's arms",
        "any comparison of I24 against exp69's committed 24-noun arms (SS6.3, "
        "doc 47 SS18.5)",
    ]
    rep["reported_not_registered_doc63_SS12_11"] = (
        "per-q and per-query-position accuracy; per-surface-order accuracy; "
        "val-minus-held-out gap per seed; selected epoch per seed; per-verb "
        "accuracy; SS11's purity traces; SS9's certificate QUANTITIES; tier (i) "
        "and tier (iii) comparator values; per-seed deltas; the (n,T) ordering. "
        "NONE MAY BE PROMOTED ONTO A CRITERION AFTER THE FACT.")
    rep["wall_time_sec_total"] = float(
        sum(v["wall_sec_total"] for v in
            list(rep["arms"].values()) + list(rep["floors"].values())))

    json.dump(rep, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== exp76 VERDICTS (doc 63 SS12), in SS18.4's reading order ===")
    for k in ("anchor_replicates_exp69", "inert_padding_control_holds",
              "comparator_flat_in_n"):
        print("  %-38s %s" % (k, verdicts[k]))
    print("  --")
    for cell in R.TRAINED_CELLS:
        print("  %-38s %s" % ("binds_%s" % cell, verdicts["binds_%s" % cell]))
    print("  --")
    for k in sorted(k2 for k2 in verdicts if k2.startswith("gap_")):
        print("  %-38s %s" % (k, verdicts[k]))
    for k in ("composition_free_at_scale", "inventory_costs",
              "grade3_under_conversion_flag", "bug_hunt_triggered",
              "entanglement_area_law_like", "mps_feasible_at_chi64"):
        print("  %-38s %s" % (k, verdicts[k]))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
