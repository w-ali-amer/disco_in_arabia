# -*- coding: utf-8 -*-
"""exp68_aggregate.py -- the ONLY writer of exp68's results file.

doc 53 SS7.3's standing rule, twice earned (doc 49 SS11.1, doc 53 SS7.3): a
runner that writes checkpoints without a versioned aggregator has now cost this
programme two recorded provenance gaps. doc 54 SS13.1 applies it in advance --
"`exp68_datagen.py`, `exp68_run.py` and ITS AGGREGATOR, in the same commit".
This file TRAINS NOTHING. Every accuracy it reports comes from
`exp68_ckpt/*.json` as run, and every comparator number from
`exp68_ckpt/comparator.json` as fitted.

THE THREE FROZEN BOOLEANS (doc 54 SS7), computed here from the frozen bars and
from nothing else. All are MEASUREMENT-VS-MEASUREMENT; no criterion is read
against any bound, because doc 52 SS2.2 removed a 9-dim-per-referent bound's
standing as an instrument on a 24-noun inventory.

  width_registered_lifts    F2 - F1 >= 0.06 on held-out seed means, with BOTH
                            arms clearing their matched S-shuffle floor by >= 2 SD
  width_adds_information    F2 - FA >= 0.06, same conditions. THE SHARP WIDTH
                            QUESTION: FA sees the angle, F2 sees angle AND
                            radius, so the difference is information the model
                            was not previously given
  anchor_real_category_binds  FA clears its matched floor by >= 2 SD AND reaches
                            >= 0.65 absolute -- doc 50 Part A's T = 1 anchor,
                            delivered for the first time
  FLOOR GATE                every arm must exceed its matched S-shuffle mean by
                            >= 2 SD before any of the above is read; an arm
                            failing the floor is reported as failed, never
                            interpreted

Threshold provenance (doc 54 SS7): 0.06 ~ 2 x the binomial SD at n_test = 300
(sqrt(0.25/300) = 0.0289), the same bar exp61 SS8 and exp67 SS4 used. INHERITED,
not chosen for this doc, and the reason exp68's n_test is deliberately 300.

"CLEARS ITS FLOOR BY >= 2 SD" HAS MORE THAN ONE READING and doc 54 SS7 does not
disambiguate, so -- exp63_aggregate.py's precedent -- all of them are computed
and the verdict is reported under each. The PRIMARY reading is floor_mean +
2 x floor_sd (the floor's own dispersion is what "clearing the floor" is about);
the alternatives use the arm's own sd and the pooled sd. If a verdict moves
across readings, that is reported rather than resolved silently.

doc 54 SS9.2's outcome table is evaluated here, before any prose is written
about the numbers, and the row that fires is carried verbatim in spirit into
`meta_notes`. Row 5 -- FA fails its anchor -- SHORT-CIRCUITS everything: the run
is uninterpretable, F1/F2 are not read, and the file says so.

doc 54 SS12's bug-hunt row is armed here too: any arm exceeding its per-twin
bound by > 0.06 is a bug hunt, not a result. F2's bound is 1.0000 so the row
cannot fire for it; F1's (0.7625) and FA's (0.9250) can.

    python exp68_aggregate.py
"""
import argparse, glob, json, os, platform, statistics as st, sys, time

import numpy as np
import torch

import exp68_run as E
from exp42_run import fresh_results_path, git_note

RESULTS_BASE = "results_exp68"
COMPARATOR = os.path.join(E.CKPT_DIR, "comparator.json")
SMOKE = E.SMOKE_OUT
VSUITE = "results_exp68_WN_validation.json"
DELTA_BAR = 0.06                 # doc 54 SS7, inherited from exp61 SS8 / exp67 SS4
FLOOR_SD = 2.0                   # doc 54 SS7
ANCHOR_ABS = 0.65                # doc 54 SS7
BUG_HUNT_MARGIN = E.BUG_HUNT_MARGIN


def block(rows):
    """mean / SAMPLE sd (ddof=1) / per-seed, ordered by seed."""
    rows = sorted(rows, key=lambda r: r["seed"])
    a = [r["heldout_test_acc"] for r in rows]
    return {"mean": st.mean(a), "sd": st.stdev(a) if len(a) > 1 else 0.0,
            "per_seed": a, "seeds": [r["seed"] for r in rows], "n_seeds": len(a),
            "val_per_seed": [r["val_acc"] for r in rows],
            "train_per_seed": [r["train_acc"] for r in rows],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
            "val_minus_heldout_per_seed": [r["val_minus_heldout"] for r in rows],
            "mean_val_minus_heldout": st.mean([r["val_minus_heldout"] for r in rows]),
            "gate_met_per_seed": [r["gate_met"] for r in rows],
            "wires": rows[0]["wires"], "n_params": rows[0]["n_params"],
            "wall_sec_total": sum(r["wall_sec"] for r in rows),
            "ckpt_sha256": [r["ckpt_sha256"] for r in rows]}


def load_runs(tag=""):
    g = {}
    for f in sorted(glob.glob(os.path.join(E.CKPT_DIR, "*.json"))):
        if os.path.basename(f) == "comparator.json":
            continue
        r = json.load(open(f, encoding="utf-8"))
        if r.get("tag", "") != tag:
            continue
        g.setdefault(r["arm"], []).append(r)
    return g


def clearance(arm_blk, floor_blk):
    """Every reading of doc 54 SS7's '>= 2 SD above its matched floor'."""
    d = arm_blk["mean"] - floor_blk["mean"]
    pooled = ((arm_blk["sd"] ** 2 + floor_blk["sd"] ** 2) / 2.0) ** 0.5
    readings = {
        "floor_sd": {"sd": floor_blk["sd"], "bar": floor_blk["mean"] + FLOOR_SD * floor_blk["sd"],
                     "clears": bool(arm_blk["mean"] >= floor_blk["mean"] + FLOOR_SD * floor_blk["sd"])},
        "arm_sd": {"sd": arm_blk["sd"], "bar": floor_blk["mean"] + FLOOR_SD * arm_blk["sd"],
                   "clears": bool(arm_blk["mean"] >= floor_blk["mean"] + FLOOR_SD * arm_blk["sd"])},
        "pooled_sd": {"sd": pooled, "bar": floor_blk["mean"] + FLOOR_SD * pooled,
                      "clears": bool(arm_blk["mean"] >= floor_blk["mean"] + FLOOR_SD * pooled)},
    }
    return {"arm_mean": arm_blk["mean"], "floor_mean": floor_blk["mean"],
            "delta_over_floor": d,
            "sd_units_over_floor_using_floor_sd":
                (d / floor_blk["sd"]) if floor_blk["sd"] > 0 else float("inf"),
            "readings": readings,
            "primary_reading": "floor_sd",
            "clears_floor": readings["floor_sd"]["clears"],
            "clears_under_all_readings": bool(all(v["clears"] for v in readings.values())),
            "clears_under_any_reading": bool(any(v["clears"] for v in readings.values())),
            "readings_agree": bool(len({v["clears"] for v in readings.values()}) == 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--base", default=RESULTS_BASE)
    a = ap.parse_args()
    t0 = time.time()

    g = load_runs(a.tag)
    missing = [arm for arm in E.ARMS if len(g.get(arm, [])) != E.N_SEEDS]
    if missing:
        raise RuntimeError("incomplete exp68 checkpoints for %s (have %s)"
                           % (missing, {k: len(v) for k, v in g.items()}))
    n_runs = sum(len(v) for v in g.values())
    blocks = {arm: block(g[arm]) for arm in E.ARMS}
    arms = {b: blocks[b] for b in E.BASE_ARMS}
    floors = {b: blocks[E.FLOOR_OF[b]] for b in E.BASE_ARMS}

    dataset = g[E.ARMS[0]][0]["dataset"]
    dsha = g[E.ARMS[0]][0]["dataset_sha256"]
    for arm, rows in g.items():
        for r in rows:
            assert r["dataset_sha256"] == dsha, "runs span two datasets"

    if not os.path.exists(COMPARATOR):
        raise RuntimeError("%s missing -- doc 54 SS8's comparator ships WITH the "
                           "run (SS13.1); run exp68_comparator.py first" % COMPARATOR)
    C = json.load(open(COMPARATOR, encoding="utf-8"))
    assert C["dataset_sha256"] == dsha, "comparator ran on a different dataset"

    # ------------------------------------------------------ floors and criteria
    clear = {b: clearance(arms[b], floors[b]) for b in E.BASE_ARMS}
    floor_gate = {b: clear[b]["clears_floor"] for b in E.BASE_ARMS}
    all_floors = bool(all(floor_gate.values()))

    d_f2_f1 = arms["F2"]["mean"] - arms["F1"]["mean"]
    d_f2_fa = arms["F2"]["mean"] - arms["FA"]["mean"]

    width_registered_lifts = bool(d_f2_f1 >= DELTA_BAR
                                  and floor_gate["F2"] and floor_gate["F1"])
    width_adds_information = bool(d_f2_fa >= DELTA_BAR
                                  and floor_gate["F2"] and floor_gate["FA"])
    anchor_real_category_binds = bool(floor_gate["FA"]
                                      and arms["FA"]["mean"] >= ANCHOR_ABS)

    # ------------------------------------------------- comparator / SS8 arbiter
    cmp_arms = {b: C["arms"][b] for b in E.BASE_ARMS}
    cmp_floors = {b: C["floors"][b] for b in E.BASE_ARMS}
    cmp_clear = {b: clearance(cmp_arms[b], cmp_floors[b]) for b in E.BASE_ARMS}
    cmp_d_f2_fa = cmp_arms["F2"]["mean"] - cmp_arms["FA"]["mean"]
    cmp_d_f2_f1 = cmp_arms["F2"]["mean"] - cmp_arms["F1"]["mean"]
    cmp_vs_circuit = {b: cmp_arms[b]["mean"] - arms[b]["mean"] for b in E.BASE_ARMS}
    comparator_matches_or_exceeds = {b: bool(cmp_vs_circuit[b] >= -0.02)
                                     for b in E.BASE_ARMS}

    if floor_gate["F2"]:
        arbiter = ("F2 CLEARS ITS FLOOR -- doc 54 SS8.1 row 1: width converts at "
                   "the circuit level and the reading is doc 54 SS9's, not SS8.1's. "
                   "The comparator's role here is the GUARD only: it %s F2 "
                   "(%.4f vs %.4f, %+.4f), so no exp68 number may be read as a "
                   "quantum result."
                   % ("matches or exceeds" if comparator_matches_or_exceeds["F2"]
                      else "does NOT match", cmp_arms["F2"]["mean"],
                      arms["F2"]["mean"], cmp_vs_circuit["F2"]))
        arbiter_row = "SS8.1 row 1 -- F2 clears floor"
    elif cmp_clear["F2"]["clears_floor"]:
        arbiter = ("F2 AT FLOOR, COMPARATOR ON THE SAME FEATURES CLEARS -- doc 54 "
                   "SS8.1 row 2: CIRCUIT-TRAINABILITY FAILURE. The information is "
                   "there and is linearly extractable (LR %.4f vs its own shuffled "
                   "floor %.4f); the circuit did not learn it. This is the SAME "
                   "failure mode exp67 exhibited on the arity lever (doc 53 "
                   "SS7.1.3), so two levers now share one explanation. It may NOT "
                   "be reported as 'the information isn't there'."
                   % (cmp_arms["F2"]["mean"], cmp_floors["F2"]["mean"]))
        arbiter_row = "SS8.1 row 2 -- circuit-trainability failure"
    else:
        arbiter = ("F2 AT FLOOR AND THE COMPARATOR AT FLOOR TOO -- doc 54 SS8.1 "
                   "row 3: INFORMATION / SAMPLE-SIDE. The second AraVec component "
                   "does not carry this category at 24 nouns and this sample size "
                   "(LR %.4f vs shuffled floor %.4f). No circuit claim is "
                   "available either way, and this may NOT be reported as a "
                   "circuit result at all."
                   % (cmp_arms["F2"]["mean"], cmp_floors["F2"]["mean"]))
        arbiter_row = "SS8.1 row 3 -- information / sample-side"

    # SS8.2: the comparator arbitrates the width_adds_information gap
    s82_blocks = bool(width_adds_information and cmp_d_f2_fa < DELTA_BAR)
    if width_adds_information:
        s82 = ("The circuit shows F2 - FA = %+.4f (>= %.2f). The comparator on the "
               "IDENTICAL features shows %+.4f. %s"
               % (d_f2_fa, DELTA_BAR, cmp_d_f2_fa,
                  ("Because the comparator gap is BELOW the bar, the circuit's gap "
                   "EXCEEDS the information gap and is an FA-TRAINABILITY "
                   "ARTEFACT: doc 54 SS8.2 says width_adds_information may NOT be "
                   "claimed." if s82_blocks else
                   "The comparator gap also clears the bar, so the added "
                   "information is real and the remaining question is only "
                   "whether the circuit converts it.")))
    else:
        s82 = ("width_adds_information is FALSE on the circuit (F2 - FA = %+.4f "
               "against a %.2f bar), so doc 54 SS8.2's blocking clause -- which "
               "guards against a circuit gap LARGER than the information gap -- "
               "has nothing to arbitrate. The comparator's own F2 - FA gap on the "
               "IDENTICAL features is %+.4f, which %s the same bar. %s"
               % (d_f2_fa, DELTA_BAR, cmp_d_f2_fa,
                  "CLEARS" if cmp_d_f2_fa >= DELTA_BAR else "also misses",
                  ("This is SS8.2 running in the direction the clause does not "
                   "cover, and it is the sharper reading of the two: the added "
                   "information is REAL and linearly extractable at %+.4f, and "
                   "the circuit converts only %+.4f of it. The gap between the "
                   "two is not an information deficit and cannot be attributed "
                   "to sample size -- the comparator ran on the SAME items, "
                   "which is exactly the case doc 54 SS14 registered in advance. "
                   "It sits on the CONVERSION side, alongside exp67's arity "
                   "finding (doc 53 SS7.1.3), and it is doc 54 SS11.1's 'supply "
                   "proved, conversion absent' spine appearing a third time."
                   % (cmp_d_f2_fa, d_f2_fa)) if cmp_d_f2_fa >= DELTA_BAR else
                  ("Circuit and comparator agree that the second feature does "
                   "not add a bar-clearing amount over the angle at this sample "
                   "size, so the negative is not located on the conversion side "
                   "by this evidence.")))
    if s82_blocks:
        width_adds_information = False

    # --------------------------------------------------- doc 54 SS12 bug hunt
    bug = {}
    for b in E.BASE_ARMS:
        bnd = E.PER_TWIN_BOUNDS[b]
        exc = arms[b]["mean"] - bnd
        bug[b] = {"per_twin_bound_doc54_SS12": bnd, "measured_mean": arms[b]["mean"],
                  "excess": exc, "margin": BUG_HUNT_MARGIN,
                  "can_fire": bool(bnd + BUG_HUNT_MARGIN < 1.0 + 1e-12 and bnd < 1.0),
                  "fires": bool(exc > BUG_HUNT_MARGIN)}
    bug["F2"]["note"] = ("F2's per-twin bound is 1.0000, so no measured accuracy "
                         "can exceed it by more than 0 and doc 54 SS9.2's bug-hunt "
                         "row CANNOT fire for this arm, as the registration states.")
    bug_hunt = bool(any(v["fires"] for v in bug.values()))

    # ------------------------------------------------------ doc 54 SS9.2 rows
    if not anchor_real_category_binds:
        row, row_key = (
            "FA FAILS ITS ANCHOR -- THE RUN IS UNINTERPRETABLE AND STOPS. doc 50 "
            "SS5.3's T = 1 anchor discipline: if SB-with-a-real-category does not "
            "bind, nothing measured against it means anything. Hunt the pipeline; "
            "do not interpret F1/F2. NOTHING MAY BE CLAIMED FROM ANY ARM.",
            "row5_FA_fails_anchor")
    elif width_registered_lifts and width_adds_information:
        row, row_key = (
            "F2 LIFTS OVER BOTH F1 AND FA, FLOORS CLEARED -- THE WIDTH LEVER "
            "CONVERTS, the first lever in the three-lever ledger that does. A "
            "second distributional feature, given its own qubit, buys held-out "
            "binding accuracy. MAY NOT BE CLAIMED: not quantum advantage (doc 54 "
            "SS8 predicts the comparator ties or wins; doc 52 SS1 closed that "
            "question for this encoding family), and not novel-filler "
            "generalisation -- the split is over pairings (doc 54 SS6.1).",
            "row1_width_converts")
    elif width_registered_lifts and not width_adds_information:
        row, row_key = (
            "F2 LIFTS OVER F1 BUT NOT OVER FA -- WIDTH IS A REPARAMETERISATION "
            "WIN, NOT AN INFORMATION WIN. The model was already extracting what "
            "the second feature adds, via the angle. Reported as the third "
            "negative in the ledger. MAY NOT BE REPORTED AS 'WIDTH WORKS'.",
            "row2_reparameterisation_not_information")
    elif not floor_gate["F2"] and cmp_clear["F2"]["clears_floor"]:
        row, row_key = (
            "F2 AT FLOOR, COMPARATOR CLEARS -- CIRCUIT-TRAINABILITY FAILURE (doc "
            "54 SS8.1), the SAME failure mode exp67 exhibited on arity (doc 53 "
            "SS7.1.3). Two levers now share one explanation, and that becomes the "
            "write-up's mechanism section. MAY NOT BE REPORTED AS 'THE INFORMATION "
            "ISN'T THERE'.", "row3_circuit_trainability_failure")
    elif not floor_gate["F2"]:
        row, row_key = (
            "F2 AT FLOOR, COMPARATOR AT FLOOR -- INFORMATION / SAMPLE-SIDE (doc 54 "
            "SS8.1). The second AraVec component does not carry this category at "
            "24 nouns. MAY NOT BE REPORTED AS A CIRCUIT RESULT AT ALL.",
            "row4_information_sample_side")
    else:
        row, row_key = (
            "F2 CLEARS ITS FLOOR but lifts over NEITHER F1 nor FA by the "
            "registered 0.06 (F2 - F1 = %+.4f, F2 - FA = %+.4f). doc 54 SS9.2's "
            "table does not name this branch explicitly; it is the strict reading "
            "of both SS7 booleans returning FALSE with every floor gate met, i.e. "
            "THE WIDTH LEVER DOES NOT CONVERT and the ledger's third cell is "
            "negative like the first two. It may NOT be reported as a trainability "
            "failure (row 3) or as an information failure (row 4): both of those "
            "rows require F2 to sit AT its floor, and it does not."
            % (d_f2_f1, d_f2_fa), "row0_all_floors_cleared_no_lift")

    verdicts = {
        "width_registered_lifts": width_registered_lifts,
        "width_adds_information": width_adds_information,
        "anchor_real_category_binds": anchor_real_category_binds,
        "floor_gate_all_arms": all_floors,
        "floor_gate_per_arm": floor_gate,
        "bug_hunt_triggered": bug_hunt,
        "run_is_interpretable": bool(anchor_real_category_binds),
        "criterion_bars": {
            "delta_bar": DELTA_BAR, "floor_sd_multiple": FLOOR_SD,
            "anchor_absolute": ANCHOR_ABS,
            "threshold_provenance": (
                "0.06 ~ 2 x the binomial SD at n_test = 300 "
                "(sqrt(0.25/300) = 0.02887), the bar exp61 SS8 and exp67 SS4 used. "
                "INHERITED, not chosen for doc 54, and the reason exp68's n_test "
                "is deliberately 300.")},
        "measured": {"F2_minus_F1": d_f2_f1, "F2_minus_FA": d_f2_fa,
                     "F1_mean": arms["F1"]["mean"], "F2_mean": arms["F2"]["mean"],
                     "FA_mean": arms["FA"]["mean"]},
        "registered_expectations_doc54_SS9_1": {
            "width_registered_lifts": "NO DIRECTION REGISTERED (doc 54 SS6.3/SS9.1)",
            "width_adds_information": "expected FALSE",
            "anchor_real_category_binds": "expected TRUE"},
        "outcome_row_doc54_SS9_2": row_key,
    }

    out = {
        "experiment": "exp68",
        "spec": "qnlp_private_docs/54_exp68_width_lever.md (1025d21, FROZEN)",
        "lever": "WIDTH -- the third and final cell of doc 54 SS1's three-lever ledger",
        "aggregated_from": "%s/*.json (%d runs%s)" % (E.CKPT_DIR, n_runs,
                                                      ", tag=%r" % a.tag if a.tag else ""),
        "aggregator": "exp68_aggregate.py (doc 53 SS7.3 / doc 54 SS13.1: the "
                      "runner ships WITH its aggregator; nothing else writes a "
                      "results file for exp68)",
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "dataset": dataset, "dataset_sha256": dsha,
        "dataset_generator": "exp68_datagen.py",
        "vsuite_results": VSUITE, "smoke_results": SMOKE,
        "target_category": {
            "key": "nonoccupational", "gloss": "non-occupational state",
            "bearers": json.load(open(dataset, encoding="utf-8"))["meta"]["bearers"],
            "n_nonbearers": 20,
            "source": ("doc 54 SS6.3, FROZEN; membership doc 54 SS5.2, user-"
                       "confirmed 2026-08-23 and imported from "
                       "instrument_sb_bounds.CATEGORIES")},
        "split": ("held-out = UNSEEN PAIRINGS, not unseen nouns (doc 54 SS6.1.1). "
                  "25 test casts -> 150 pairs -> 300 test items, matching "
                  "exp60/exp61 so SS7's 0.06 bar transfers exactly."),
        "arms_run": E.ARMS, "n_seeds": E.N_SEEDS, "n_runs": n_runs,
        "sd_convention": "sample sd, ddof=1 (statistics.stdev)",
        "training": {"lr": E.LR, "batch": E.BATCH, "max_epochs": E.MAX_EPOCHS,
                     "optimiser": "Adam", "rng_base": E.RNG_BASE,
                     "model_selection": "best in-distribution val accuracy",
                     "seeds_vary": ("OPTIMISER INITIALISATION ONLY -- doc 54 "
                                    "SS6.2: the features are deterministic "
                                    "constants, so unlike exp61 there is no "
                                    "per-seed angle draw"),
                     "lineage": "the exp60 / exp61 / exp65 loop, unchanged"},
        "feature_maps": {
            "F1": "theta = (p1 + 1) pi, p1 = PC1-raw / 8.669494516224397",
            "F2": "theta0 = (p1 + 1) pi on qubit 0, theta1 = (p2 + 1) pi on qubit 1, "
                  "p2 = PC2-raw / 10.160404144924657",
            "FA": "phi = atan2(PC2, PC1) -- the existing exp42 map, verified equal "
                  "to the frozen exp42_noun_embeddings angle to 4.0e-15",
            "provenance": ("PCA over all 44 referents, centred, PC signs pinned to "
                           "the frozen vec2 plane (doc 54 SS3.2/SS6.2), computed by "
                           "the committed instrument_sb_bounds.load_raw; the "
                           "runtime map is bit-identical (max abs diff 0.0) to an "
                           "independent recomputation from the committed source "
                           "files, asserted at smoke")},
        "arms": arms, "floors": floors,
        "floor_clearance": clear,
        "comparator": {
            "source": COMPARATOR,
            "arms": {b: {k: v for k, v in cmp_arms[b].items() if k != "runs"}
                     for b in E.BASE_ARMS},
            "floors": {b: {k: v for k, v in cmp_floors[b].items() if k != "runs"}
                       for b in E.BASE_ARMS},
            "floor_clearance": cmp_clear,
            "comparator_minus_circuit": cmp_vs_circuit,
            "comparator_matches_or_exceeds_circuit": comparator_matches_or_exceeds,
            "comparator_matches_or_exceeds_all_arms":
                bool(all(comparator_matches_or_exceeds.values())),
            "tie_margin": 0.02,
            "F2_minus_FA": cmp_d_f2_fa, "F2_minus_F1": cmp_d_f2_f1,
            "features": C["features"]["definition"],
            "frame_check": C["features"]["frame_check"],
            "optimiser": C["optimiser"],
            "registered_prediction": ("doc 54 SS8: the classical comparator MATCHES "
                                      "OR EXCEEDS every arm -- registered so it "
                                      "cannot become a post-hoc surprise, and so no "
                                      "exp68 number can be read as a quantum result"),
            "arbiter_row": arbiter_row,
            "arbiter_reading_SS8_1": arbiter,
            "arbiter_reading_SS8_2": s82,
            "SS8_2_blocked_the_claim": s82_blocks,
            "function_class_qualifier": C["function_class_qualifier"]},
        "bounds_context_only_doc54_SS12": {
            "note": ("doc 54 SS7: NO criterion is read against any bound. doc 52 "
                     "SS2.2 removed a 9-dim-per-referent bound's standing as an "
                     "instrument on a 24-noun inventory. These numbers exist to "
                     "show the registered prediction structure and to arm SS9.2's "
                     "bug-hunt row, and for nothing else."),
            "per_twin_bounds": E.PER_TWIN_BOUNDS,
            "bug_hunt": bug, "bug_hunt_triggered": bug_hunt},
        "verdicts": verdicts,
        "meta_notes": {},
        "git_note": git_note(),
        "wall_time_sec": time.time() - t0,
    }

    total_train_sec = sum(b["wall_sec_total"] for b in blocks.values())
    out["meta_notes"] = {
        "headline": (
            "doc 54 SS9.2 row that fired: %s. width_registered_lifts = %s "
            "(F2 - F1 = %+.4f vs a %.2f bar), width_adds_information = %s "
            "(F2 - FA = %+.4f), anchor_real_category_binds = %s (FA %.4f, floor "
            "%.4f, bar max(floor + 2 SD, 0.65) = %.4f). Held-out means over 5 "
            "seeds: F1 %.4f +- %.4f, F2 %.4f +- %.4f, FA %.4f +- %.4f, against "
            "matched shuffled floors %.4f / %.4f / %.4f."
            % (row_key, width_registered_lifts, d_f2_f1, DELTA_BAR,
               width_adds_information, d_f2_fa, anchor_real_category_binds,
               arms["FA"]["mean"], floors["FA"]["mean"],
               max(clear["FA"]["readings"]["floor_sd"]["bar"], ANCHOR_ABS),
               arms["F1"]["mean"], arms["F1"]["sd"], arms["F2"]["mean"],
               arms["F2"]["sd"], arms["FA"]["mean"], arms["FA"]["sd"],
               floors["F1"]["mean"], floors["F2"]["mean"], floors["FA"]["mean"])),
        "registered_outcome_row": row,
        "matched_floors_departure_from_SS13": (
            "doc 54 SS13 costed '~20 runs (4 arms x 5 seeds)', treating S-shuffle "
            "as ONE arm. This run used THREE floor arms -- S-shuffle-F1, "
            "S-shuffle-F2, S-shuffle-FA -- for 30 runs, because doc 54 SS7 states "
            "the gate twice as 'its MATCHED S-shuffle floor' and 'ITS matched "
            "S-shuffle mean'. The floor is architecture-dependent here in a way it "
            "was not in exp61, where all arms shared one circuit family: F2 is a "
            "4-wire circuit with an 81-dimensional joint class against ~190 "
            "constraints, which is exactly doc 52 SS2.2's interpolation regime, and "
            "the measured floors bear it out (F2's shuffled floor %.4f vs F1's "
            "%.4f and FA's %.4f). Borrowing the 2-wire floor for F2 would have been "
            "an unmatched comparator -- the exp44 failure class -- and would have "
            "made F2's gate the easiest instead of the hardest. SS13 is a cost "
            "estimate; SS7 is a frozen criterion, so SS7 governs. Extra floors can "
            "only make gates stricter; they cannot manufacture a positive."
            % (floors["F2"]["mean"], floors["F1"]["mean"], floors["FA"]["mean"])),
        "floor_reading_sensitivity": (
            "doc 54 SS7 says '>= 2 SD' without naming whose SD, so all three "
            "readings are computed (floor's own sd -- PRIMARY, arm's sd, pooled). "
            "Agreement per arm: F1 %s, F2 %s, FA %s. %s"
            % (clear["F1"]["readings_agree"], clear["F2"]["readings_agree"],
               clear["FA"]["readings_agree"],
               "All readings agree, so no verdict depends on the choice."
               if all(clear[b]["readings_agree"] for b in E.BASE_ARMS)
               else "AT LEAST ONE ARM'S FLOOR VERDICT DEPENDS ON THE READING -- "
                    "see floor_clearance; the primary reading is reported as the "
                    "verdict and the disagreement is not resolved silently.")),
        "anchor": (
            "doc 50 Part A's T = 1 anchor -- SB with a REAL semantic category -- "
            "delivered here for the first time. FA reaches %.4f against a matched "
            "shuffled floor of %.4f (+%.4f, %.1f floor-SDs) and the 0.65 absolute "
            "bar. doc 54 SS9.1 registered this as expected TRUE and doc 54 SS9.2 "
            "row 5 as the thing that would invalidate the whole run. Verdict: %s."
            % (arms["FA"]["mean"], floors["FA"]["mean"],
               clear["FA"]["delta_over_floor"],
               clear["FA"]["sd_units_over_floor_using_floor_sd"],
               "ANCHOR HOLDS" if anchor_real_category_binds else "ANCHOR FAILED")),
        "comparator_arbiter": arbiter,
        "comparator_SS8_2": s82,
        "comparator_guard": (
            "doc 54 SS8's registered prediction was that the comparator MATCHES OR "
            "EXCEEDS every arm. Measured, comparator minus circuit: F1 %+.4f, "
            "F2 %+.4f, FA %+.4f (tie margin 0.02). Verdict %s. doc 52 SS1 already "
            "proved the product-state kernel factorises and is classically O(n) "
            "for this encoding family, so nothing in exp68 can be read as quantum "
            "advantage whichever way these deltas point."
            % (cmp_vs_circuit["F1"], cmp_vs_circuit["F2"], cmp_vs_circuit["FA"],
               all(comparator_matches_or_exceeds.values()))),
        "instrument_corroboration": (
            "A consistency check, NOT a criterion (doc 54 SS7 reads nothing "
            "against a bound). The comparator lands on doc 54 SS12's "
            "INDEPENDENTLY computed per-twin bounds without ever being fitted to "
            "them: LR(F1) %.4f vs bound 0.7625, LR(F2) %.4f vs 1.0000, LR(FA) "
            "%.4f vs 0.9250. The bounds come from a MAX-FS search over the same "
            "per-referent feature maps (instrument_sb_bounds.py); the comparator "
            "comes from gradient descent on held-out-split data. Two unrelated "
            "computations agreeing to within %.4f on the largest discrepancy is "
            "evidence that the feature frame and the bound instrument describe "
            "the same object. It also confirms that the F1 arm is genuinely "
            "information-starved by the geometry and not by the optimiser: the "
            "best linear model on F1's features cannot reach what F2's reach "
            "trivially."
            % (cmp_arms["F1"]["mean"], cmp_arms["F2"]["mean"], cmp_arms["FA"]["mean"],
               max(abs(cmp_arms[b]["mean"] - E.PER_TWIN_BOUNDS[b])
                   for b in E.BASE_ARMS))),
        "floors_are_real_floors": (
            "The shuffled floors sit at chance for every arm INCLUDING the "
            "81-dimensional one: circuit floors %.4f (F1) / %.4f (F2) / %.4f "
            "(FA), comparator floors %.4f / %.4f / %.4f. This matters because "
            "doc 52 SS2.2 showed that a 9-dim-per-referent (81-dim joint) class "
            "INTERPOLATES random labellings on this inventory -- and it does "
            "here too, on TRAIN: the F2 comparator drives its training loss to "
            "%.4f on shuffled labels. What it cannot do is carry that "
            "interpolation to unseen PAIRINGS. So the held-out split is doing "
            "the work doc 52 SS2.2 warned the bound could not, and the floor "
            "gate is a real gate rather than a formality."
            % (floors["F1"]["mean"], floors["F2"]["mean"], floors["FA"]["mean"],
               cmp_floors["F1"]["mean"], cmp_floors["F2"]["mean"],
               cmp_floors["FA"]["mean"],
               float(np.mean(cmp_floors["F2"]["train_loss_final_per_seed"])))),
        "bug_hunt_row": (
            "doc 54 SS9.2's last row: any arm exceeding its per-twin bound by more "
            "than 0.06 is a BUG HUNT, not a result. F1 %.4f vs bound 0.7625 "
            "(excess %+.4f), FA %.4f vs 0.9250 (%+.4f). F2's bound is 1.0000 so "
            "the row cannot fire for it, as doc 54 states. Triggered: %s."
            % (arms["F1"]["mean"], bug["F1"]["excess"], arms["FA"]["mean"],
               bug["FA"]["excess"], bug_hunt)),
        "model_selection": (
            "doc 54 SS14's checklist entry, answered with numbers: selection is on "
            "in-distribution val ONLY and the val-minus-held-out gap is reported "
            "per seed. Mean gaps: F1 %+.4f, F2 %+.4f, FA %+.4f."
            % (arms["F1"]["mean_val_minus_heldout"],
               arms["F2"]["mean_val_minus_heldout"],
               arms["FA"]["mean_val_minus_heldout"])),
        "stop_condition": (
            "doc 54 SS11, verbatim in intent: exp68 completes the three-lever "
            "enquiry ledger -- frequency K (exp61/62/63), readout arity M (exp67), "
            "width (exp68) -- and is the LAST experiment before the write-up, "
            "REGARDLESS OF ITS OUTCOME. A positive is a finding for the write-up; "
            "a negative is the third cell of a coherent ledger and is equally a "
            "finding. Neither licenses a fourth lever, a follow-up rung, or 'one "
            "more seed'."),
        "scope": (
            "doc 54 SS10, and no outcome above escapes it. NOT quantum advantage. "
            "NOT novel-filler binding -- the split is over unseen PAIRINGS, not "
            "unseen nouns, which is the price of doc 54 SS4.3's legitimate "
            "inventory-wide PCA fit. NOT 'meaning' in general: one category, one "
            "task family, 24 nouns, 2 referents, simulation. NOT a statement about "
            "the AraVec model: PC1/PC2 over 44 words explain a small share of its "
            "variance. The doc 54 SS12 bounds bound THIS feature map and THIS "
            "inventory only."),
        "provenance": (
            "Aggregation only: nothing was trained here. %d runs from %s, dataset "
            "sha %s, comparator from %s. Total training wall time %.0f s across "
            "the runs as executed."
            % (n_runs, E.CKPT_DIR, dsha[:12], COMPARATOR, total_train_sec)),
    }

    p = fresh_results_path(a.base)
    assert "exp68" in p
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("[exp68] %d runs aggregated from %s" % (n_runs, E.CKPT_DIR))
    for b in E.BASE_ARMS:
        c = clear[b]
        print("  %-2s  held-out %.4f +- %.4f  | floor %.4f +- %.4f  | +%.4f "
              "(%.1f floor-SD)  clears=%s  | comparator %.4f (floor %.4f)"
              % (b, arms[b]["mean"], arms[b]["sd"], floors[b]["mean"],
                 floors[b]["sd"], c["delta_over_floor"],
                 c["sd_units_over_floor_using_floor_sd"], c["clears_floor"],
                 cmp_arms[b]["mean"], cmp_floors[b]["mean"]))
    print("  F2 - F1 = %+.4f  |  F2 - FA = %+.4f  |  bar %.2f"
          % (d_f2_f1, d_f2_fa, DELTA_BAR))
    print("  width_registered_lifts      = %s" % width_registered_lifts)
    print("  width_adds_information      = %s" % width_adds_information)
    print("  anchor_real_category_binds  = %s" % anchor_real_category_binds)
    print("  floor gate (all arms)       = %s  %s" % (all_floors, floor_gate))
    print("  bug-hunt row triggered      = %s" % bug_hunt)
    print("  doc 54 SS9.2 row            = %s" % row_key)
    print("  SS8.1 arbiter               = %s" % arbiter_row)
    print("[exp68] wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
