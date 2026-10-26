# -*- coding: utf-8 -*-
"""exp69_aggregate.py -- doc 55: the ONLY writer of `results_exp69.json`.

doc 53 SS7.3's rule, applied: a runner that writes checkpoints ships with the
aggregator that writes the results JSON, in the same commit. Nothing else in exp69
writes a `results_exp69*.json` except this file and `exp69_run.py --a0-gate` (which
writes `results_exp69_A0.json`, separate by Rule 3 so no exp68 artefact is touched).

WHAT IT COMPUTES

  doc 55 SS6   the frozen booleans, ALL measurement-vs-measurement, none read
               against any bound (SS12; doc 54 SS17.7 is the standing reason -- a
               bound moved by 0.0375 AFTER exp68 reported and no stored boolean
               moved with it):
                 SS6.1 anchor_replicates_exp68          BLOCKING, read first
                 SS6.2 t1_surface_anchor_binds
                 SS6.3 t2_binding_holds                 the composition criterion
                 SS6.4 t2_binding_holds_angle
                 SS6.5 register_beats_angle_at_T2 / _at_T1
                 SS6.6 floor gate, ALL THREE SD readings, per arm
                 SS6.7 comparator_matches_or_exceeds_all_arms
                 SS6.8 bug_hunt_triggered
               plus SS18.4's Part B criteria wazn_binds_T1 / wazn_binds_T2.
  doc 55 SS18.2  composition costs cc_A / cc_W / cc_M and the interaction dcc,
               DESCRIPTIVE, per-seed values shown, exp44 no-promotion clause stated.
  doc 55 SS19.2  the prediction-vs-measured table, WITH ITS GRADES KEPT APART.
  doc 55 SS19.3  Spearman(witness margin, per-item held-out accuracy), per arm.
  doc 55 SS19.4  the purity contrast, stratified by q AND by correctness.
  doc 55 SS19.5  error overlap (Cohen's kappa + the two error-subset fractions) and
               primacy/recency.
  doc 55 SS6.9   REPORTED, NOT REGISTERED: per-q accuracy, per-surface-order
               accuracy, the val-heldout selection gap, per-verb-pair accuracy.
               NONE of these may be promoted onto a criterion after the fact --
               that is Rule 4's second sentence and the exp44 lesson.
  doc 55 SS9.2   which outcome rows fired, quoted from the doc, chosen by the
               booleans and not by inspection.

    python exp69_aggregate.py
"""
import argparse
import glob
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch

import exp42_compiler as comp
import exp69_run as R

SPEC = "qnlp_private_docs/55_exp69_reunion.md (da4b8fe, REGISTERED through SS19.7)"
SPEC_COMMIT = "da4b8fe"
RESULTS = "results_exp69.json"
COMPARATOR = os.path.join(R.CKPT_DIR, "comparator.json")
SMOKE = R.SMOKE_OUT
VSUITE = "results_exp69_WN_validation.json"
A0_FILE = R.A0_OUT
PREDICTIONS = R.PREDICTIONS

DELTA_BAR = 0.06        # doc 55 SS6.5, inherited: 2 x binomial SD at n_test = 300
FLOOR_SD = 2.0          # doc 55 SS6.6
ANCHOR_ABS = 0.65       # doc 55 SS6.3, inherited verbatim from doc 54 SS7
ANCHOR_BAND = 0.06      # doc 55 SS6.2
ANCHOR_POINT = 0.9187   # doc 55 SS6.2, exp68's FA held-out mean
TIE_BAND = 0.02         # doc 55 SS6.7
BUG_HUNT_MARGIN = R.BUG_HUNT_MARGIN


# ------------------------------------------------------------------- helpers
def block(rows):
    rows = sorted(rows, key=lambda r: r["seed"])
    a = [r["heldout_test_acc"] for r in rows]
    return {"mean": st.mean(a), "sd": st.stdev(a) if len(a) > 1 else 0.0,
            "per_seed": a, "seeds": [r["seed"] for r in rows], "n_seeds": len(a),
            "T": rows[0]["T"], "part": rows[0]["part"],
            "encoding": rows[0]["encoding"], "role": rows[0]["role"],
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
    for f in sorted(glob.glob(os.path.join(R.CKPT_DIR, "*.json"))):
        if os.path.basename(f) == "comparator.json":
            continue
        r = json.load(open(f, encoding="utf-8"))
        if r.get("tag", "") != tag:
            continue
        g.setdefault(r["arm"], []).append(r)
    return g


def clearance(arm_blk, floor_blk):
    """doc 55 SS6.6: EVERY reading of '>= 2 SD above its matched floor'. An arm whose
    readings disagree is reported as DISAGREEING rather than resolved silently."""
    d = arm_blk["mean"] - floor_blk["mean"]
    pooled = ((arm_blk["sd"] ** 2 + floor_blk["sd"] ** 2) / 2.0) ** 0.5
    readings = {
        "floor_sd": {"sd": floor_blk["sd"],
                     "bar": floor_blk["mean"] + FLOOR_SD * floor_blk["sd"],
                     "clears": bool(arm_blk["mean"]
                                    >= floor_blk["mean"] + FLOOR_SD * floor_blk["sd"])},
        "arm_sd": {"sd": arm_blk["sd"],
                   "bar": floor_blk["mean"] + FLOOR_SD * arm_blk["sd"],
                   "clears": bool(arm_blk["mean"]
                                  >= floor_blk["mean"] + FLOOR_SD * arm_blk["sd"])},
        "pooled_sd": {"sd": pooled, "bar": floor_blk["mean"] + FLOOR_SD * pooled,
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
            "readings_agree": bool(len({v["clears"]
                                        for v in readings.values()}) == 1)}


def spearman(x, y):
    """Rank correlation with midranks for ties (the margins are cast-level, so
    ties are structural and must be handled, not ignored)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1, dtype=float)
        # midranks
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = 0.0
    for c in (0, 1):
        pe += float((a == c).mean()) * float((b == c).mean())
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else None


def strat(rows, keyfn):
    cell = {}
    for r in rows:
        k = keyfn(r)
        cell.setdefault(k, [0, 0])
        cell[k][0] += int(r["correct"])
        cell[k][1] += 1
    return {str(k): {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(cell.items())}


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args()
    t0 = time.time()

    g = load_runs(a.tag)
    missing = [arm for arm in R.ARMS if len(g.get(arm, [])) != R.N_SEEDS]
    if missing:
        raise RuntimeError("incomplete exp69 checkpoints for %s (have %s)"
                           % (missing, {k: len(v) for k, v in g.items()}))
    n_runs = sum(len(v) for v in g.values())
    blocks = {arm: block(g[arm]) for arm in R.ARMS}
    arms = {b: blocks[b] for b in R.BASE_ARMS}
    floors = {b: blocks[R.FLOOR_OF[b]] for b in R.FLOORED}

    for f in (COMPARATOR, PREDICTIONS, A0_FILE):
        if not os.path.exists(f):
            raise RuntimeError("%s missing -- doc 55 SS8.5 / SS19.2 / SS6.1 require it "
                               "before any result is read" % f)
    C = json.load(open(COMPARATOR, encoding="utf-8"))
    P = json.load(open(PREDICTIONS, encoding="utf-8"))
    A0 = json.load(open(A0_FILE, encoding="utf-8"))

    # ---- datasets, for the structural join the item-level analyses need
    ds = {}
    item_of = {}
    for part, path in (("A", R.DATA_A), ("B", R.DATA_B), ("exp68", R.DATA_EXP68)):
        blob = json.load(open(path, encoding="utf-8"))
        assert comp.canonical_sha256(blob["items"]) == blob["meta"]["dataset_sha256"]
        ds[part] = {"path": path, "sha256": blob["meta"]["dataset_sha256"],
                    "meta": {k: v for k, v in blob["meta"].items() if k != "items"}}
        # KEYED BY (part, id). Part A and Part B use the SAME id scheme
        # ("train_p1_00001_a", ...) because pid1/pid2 restart per part, so a flat
        # id map would let Part B silently overwrite Part A -- and the two parts
        # carry DIFFERENT V_A/V_B partitions, so every item_of-derived stratum in
        # SS6.9 would be misattributed. Not hypothetical; guaranteed.
        for it in blob["items"]:
            item_of[(part, it["id"])] = it
    for arm, rows in g.items():
        for r in rows:
            assert r["dataset_sha256"] == ds[r["part"]]["sha256"], \
                "run %s/%d ran on a different dataset than the one on disk" % (arm, r["seed"])

    # ============================================== doc 55 SS6.1, READ FIRST
    anchor_replicates_exp68 = bool(A0["anchor_replicates_exp68"])

    # ============================================== doc 55 SS6.6, the floor gate
    clear = {b: clearance(arms[b], floors[b]) for b in R.FLOORED}
    floor_gate = {b: clear[b]["clears_floor"] for b in R.FLOORED}
    readings_disagree = [b for b in R.FLOORED if not clear[b]["readings_agree"]]
    all_floors = bool(all(floor_gate.values()))

    # ============================================== the comparator, SS8 / SS6.7
    cmp_arms = {b: C["arms"][b] for b in R.BASE_ARMS}
    cmp_floors = {b: C["floors"][b] for b in R.FLOORED}
    cmp_clear = {b: clearance(cmp_arms[b], cmp_floors[b]) for b in R.FLOORED}
    cmp_vs_circuit = {b: cmp_arms[b]["mean"] - arms[b]["mean"] for b in R.BASE_ARMS}
    cmp_ge = {b: bool(cmp_vs_circuit[b] >= -TIE_BAND) for b in R.BASE_ARMS}
    comparator_matches_or_exceeds_all_arms = bool(all(cmp_ge.values()))

    # ============================================== doc 55 SS6.2 - SS6.5
    m = {b: arms[b]["mean"] for b in R.BASE_ARMS}
    t1_surface_anchor_binds = bool(
        floor_gate["R1-A"] and abs(m["R1-A"] - ANCHOR_POINT) <= ANCHOR_BAND)
    t2_binding_holds = bool(floor_gate["R2-W"] and m["R2-W"] >= ANCHOR_ABS)
    t2_binding_holds_angle = bool(floor_gate["R2-A"] and m["R2-A"] >= ANCHOR_ABS)
    wazn_binds_T1 = bool(floor_gate["R1-M"] and m["R1-M"] >= ANCHOR_ABS)
    wazn_binds_T2 = bool(floor_gate["R2-M"] and m["R2-M"] >= ANCHOR_ABS)

    d_T2 = m["R2-W"] - m["R2-A"]
    d_T1 = m["R1-W"] - m["R1-A"]
    cmp_d_T2 = cmp_arms["R2-W"]["mean"] - cmp_arms["R2-A"]["mean"]
    cmp_d_T1 = cmp_arms["R1-W"]["mean"] - cmp_arms["R1-A"]["mean"]
    # SS8.3: if the circuit gap clears the bar but the comparator gap does not, the
    # circuit's gap EXCEEDS the information gap and is an angle-trainability artefact
    s83_blocks_T2 = bool(d_T2 >= DELTA_BAR and cmp_d_T2 < DELTA_BAR)
    s83_blocks_T1 = bool(d_T1 >= DELTA_BAR and cmp_d_T1 < DELTA_BAR)
    register_beats_angle_at_T2 = bool(
        d_T2 >= DELTA_BAR and floor_gate["R2-W"] and floor_gate["R2-A"]
        and not s83_blocks_T2)
    register_beats_angle_at_T1 = bool(
        d_T1 >= DELTA_BAR and floor_gate["R1-W"] and floor_gate["R1-A"]
        and not s83_blocks_T1)

    # ============================================== doc 55 SS6.8, the bug hunt
    bounds = {b: P["grade1_bounds"][R.BOUND_FAMILY[b]]["bound_value"]
              for b in R.BASE_ARMS}
    over = {b: m[b] - bounds[b] for b in R.BASE_ARMS}
    bug_hunt = {b: bool(over[b] > BUG_HUNT_MARGIN) for b in R.BASE_ARMS}
    bug_hunt_triggered = bool(any(bug_hunt.values()))

    # ================================= doc 55 SS18.2, DESCRIPTIVE composition cost
    def cc(t1, t2):
        per = [x - y for x, y in zip(arms[t1]["per_seed"], arms[t2]["per_seed"])]
        return {"pair": "%s - %s" % (t1, t2),
                "mean": arms[t1]["mean"] - arms[t2]["mean"],
                "per_seed": per, "sd": st.stdev(per) if len(per) > 1 else 0.0}
    cc_A, cc_W, cc_M = cc("R1-A", "R2-A"), cc("R1-W", "R2-W"), cc("R1-M", "R2-M")
    dcc = {"definition": "dcc = cc_A - cc_W",
           "mean": cc_A["mean"] - cc_W["mean"],
           "per_seed": [x - y for x, y in zip(cc_A["per_seed"], cc_W["per_seed"])]}
    composition = {
        "cc_A": cc_A, "cc_W": cc_W, "cc_M": cc_M, "interaction_dcc": dcc,
        "status": "DESCRIPTIVE ONLY -- NEITHER IS A CRITERION",
        "no_promotion_clause": (
            "doc 55 SS18.2, stated before results existed: neither cc nor dcc may be "
            "PROMOTED ONTO A CRITERION after results exist. At 5 seeds both are "
            "knife-edge magnets, which is why the per-seed values are printed "
            "alongside every mean. doc 55 SS18.2 registers NO CONFIDENT DIRECTION for "
            "dcc: the precedents cut both ways (richer parameterisation trained worse "
            "twice -- exp62 S2 < S1, exp67 R8 < R2 at M2 -- yet the register "
            "converted at T = 1 in exp68), so both branches are readable and neither "
            "is a surprise."),
        "population_note": (
            "doc 55 SS6.10(b) refuses any T = 1 vs T = 2 accuracy DELTA as a "
            "CRITERION, because R1 and R2 are different tasks with different label "
            "distributions and differencing them is doc 47 SS18.5's population "
            "mismatch. SS18.2 registers the same difference as a DESCRIPTIVE "
            "quantity, and the distinction is exactly that: reported, never read."),
    }

    # =================================== doc 55 SS19.2, prediction vs measured
    pred_table = {}
    for b in R.BASE_ARMS:
        fam = R.BOUND_FAMILY[b]
        g2 = P["grade2_point_predictions"]["by_arm"][b]
        pred_table[b] = {
            "family": fam,
            "grade1_bound": bounds[b],
            "grade1_method": ("EXACT" if P["grade1_bounds"][fam]["exact"]
                              else "CONSTRUCTIVE LOWER BOUND"),
            "grade2_point": g2["point_prediction"],
            "grade2_interval": g2["interval"],
            "measured": m[b],
            "measured_minus_point": m[b] - g2["point_prediction"],
            "inside_grade2_interval": bool(g2["interval"][0] <= m[b]
                                           <= g2["interval"][1]),
            "conversion_ratio_measured_over_bound": m[b] / bounds[b],
            "over_bound_by": over[b],
            "bug_hunt_row_fires": bug_hunt[b],
        }
    n_in = sum(1 for v in pred_table.values() if v["inside_grade2_interval"])
    predictions_report = {
        "grades_kept_apart": (
            "doc 55 SS19.2. GRADE 1 is the BOUND, computed by the committed "
            "instrument before datagen and frozen in exp69_predictions.json. GRADE 2 "
            "is the CONVERSION POINT PREDICTION, bound x the trained-population "
            "single-RY conversion median (%.4f, n = %d rows in ONE task family), "
            "with a +/- %.2f interval, AND THE EXTRAPOLATION IS ITSELF UNDER TEST -- "
            "exp69 is a NEW task family, so a miss refutes the EXTRAPOLATION and not "
            "the bound. GRADE 3 was computed before freezing and recorded as "
            "UNTESTABLE this run: no exp69 arm has a below-null category."
            % (P["grade2_point_predictions"]["conversion_median"],
               P["grade2_point_predictions"]["conversion_median_n_rows"],
               DELTA_BAR)),
        "grade3": P["grade3_under_conversion_flag"],
        "table": pred_table,
        "n_arms_inside_grade2_interval": n_in, "n_arms": len(pred_table),
        "extrapolation_verdict": (
            "the grade-2 extrapolation HOLDS on %d of %d arms" % (n_in, len(pred_table))
            if n_in == len(pred_table) else
            "the grade-2 extrapolation MISSES on %d of %d arms. doc 55 SS19.2: a miss "
            "refutes THE EXTRAPOLATION (a conversion median from n = %d rows in one "
            "task family, carried into a new family), NOT the bound, and NOT the "
            "calculus. No criterion in doc 55 SS6 reads any of these numbers."
            % (len(pred_table) - n_in, len(pred_table),
               P["grade2_point_predictions"]["conversion_median_n_rows"])),
        "bounds_are_context_only": P["bounds_are_context_only"],
    }

    # ============ item-level machinery for SS19.3 / SS19.4 / SS19.5 and SS6.9
    def per_item(arm):
        """{item_id: [correct per seed]} plus one representative row per item."""
        acc, rep = {}, {}
        for r in sorted(g[arm], key=lambda x: x["seed"]):
            for row in r["heldout_items"]:
                acc.setdefault(row["id"], []).append(int(row["correct"]))
                rep.setdefault(row["id"], row)
        return acc, rep

    W = P["witnesses_doc55_SS19_3"]
    s193, s194, s195, s69 = {}, {}, {}, {}
    for b in R.BASE_ARMS:
        acc, rep = per_item(b)
        fam = R.BOUND_FAMILY[b]
        # ---- SS19.3 Spearman(margin, per-item held-out accuracy)
        tw = W[fam]["per_twin"]
        xs, ys, miss = [], [], 0
        for iid, ks in acc.items():
            key = "%s|%s" % (rep[iid]["agent"], rep[iid]["patient"])
            if key not in tw:
                miss += 1
                continue
            xs.append(tw[key])
            ys.append(float(np.mean(ks)))
        assert miss == 0, \
            ("SS19.3 margin join broke for %s: %d held-out items have no frozen "
             "witness margin. A nonzero count means the (agent, patient) keys do "
             "not index the twin design the bound was computed over -- loud, not a "
             "field." % (b, miss))
        rho = spearman(xs, ys)
        # cast-level, where the margin actually varies
        bycast = {}
        for iid, ks in acc.items():
            key = "%s|%s" % (rep[iid]["agent"], rep[iid]["patient"])
            if key in tw:
                bycast.setdefault(key, []).append(float(np.mean(ks)))
        rho_cast = spearman([tw[k] for k in bycast],
                            [float(np.mean(v)) for v in bycast.values()])
        s193[b] = {
            "witness_family": fam, "n_items": len(xs), "n_unmatched": miss,
            "n_distinct_margins": len(set(xs)),
            "spearman_item_level": rho, "spearman_cast_level": rho_cast,
            "n_casts": len(bycast),
            "direction_registered": "positive expected (doc 55 SS19.3)",
            "status": "DESCRIPTIVE -- no criterion reads it",
            "resolution_note": (
                "the witness margin is a function of the (agent, patient) ORDERED "
                "PAIR alone, so all items sharing a queried-event direction on the "
                "same cast share ONE margin: %d distinct margin values over %d "
                "held-out items (%d casts). doc 55 SS19.3's '150 pairs per arm' "
                "counts test PAIRS; the margin's own resolution is cast-level, and "
                "both readings are given rather than the coarser one being hidden. "
                "Ties are handled by midranks."
                % (len(set(xs)), len(xs), len(bycast))),
            "caveat": W["registered_caveat"] if fam == "R-W" else None,
        }
        # ---- SS19.4 purity contrast, stratified by q and correctness
        cells = {}
        for iid, row in rep.items():
            q = row["query_index"]
            ok = bool(np.mean(acc[iid]) >= 0.5)
            pur = row["purity"]
            k = "q%d_%s" % (q, "correct" if ok else "incorrect")
            c = cells.setdefault(k, {"n": 0, "stages": {}})
            c["n"] += 1
            for stage, vals in pur.items():
                c["stages"].setdefault(stage, []).append(float(np.mean(vals)))
        cell_out = {}
        for k, c in sorted(cells.items()):
            cell_out[k] = {"n": c["n"],
                           "mean_wire_purity": {s: float(np.mean(v))
                                                for s, v in sorted(c["stages"].items())}}
        # the SS19.4 restoration signature: |readout - after_event1|
        restore = {}
        for q in (1, 2):
            vals = [abs(float(np.mean(rep[i]["purity"]["at_readout"]))
                        - float(np.mean(rep[i]["purity"]["after_event1"])))
                    for i in rep if rep[i]["query_index"] == q
                    and "after_event1" in rep[i]["purity"]]
            if vals:
                restore["q%d" % q] = {"n": len(vals), "mean_abs_gap": float(np.mean(vals)),
                                      "max_abs_gap": float(np.max(vals))}
        s194[b] = {
            "T": R.ARM_CFG[b]["T"], "cells": cell_out,
            "readout_minus_after_event1_abs": restore,
            "registered_signature": (
                "doc 55 SS19.4: the T = 1 undo-signature (readout returns every wire "
                "to purity) does NOT transfer to T = 2 as-is, because the queried "
                "block's inverse acts AFTER the other event has entangled the wires "
                "and the operations do not commute. The registered expectation is "
                "therefore ASYMMETRIC IN QUERY POSITION: for q = 2 the inverse undoes "
                "the queried block exactly and readout-stage purity returns to its "
                "post-event-1 level; for q = 1 only a partial restoration is "
                "expected. DESCRIPTIVE; no criterion reads it."),
            "status": "DESCRIPTIVE -- doc 55 SS18.3 / SS19.4",
        }
        # ---- SS19.5(a) error overlap with the comparator, per seed
        krows, subs = [], []
        for cr, mr in zip(sorted(g[b], key=lambda x: x["seed"]),
                          sorted(C["arms"][b]["runs"], key=lambda x: x["seed"])):
            ca = {r["id"]: r for r in cr["heldout_items"]}
            ma = {r["id"]: r for r in mr["heldout_items"]}
            ids = sorted(set(ca) & set(ma))
            kp = cohen_kappa([ca[i]["pred"] for i in ids], [ma[i]["pred"] for i in ids])
            ce = {i for i in ids if not ca[i]["correct"]}
            me = {i for i in ids if not ma[i]["correct"]}
            krows.append(kp)
            subs.append({"seed": cr["seed"], "n": len(ids),
                         "n_circuit_errors": len(ce), "n_comparator_errors": len(me),
                         "circuit_errors_that_are_also_comparator_errors":
                             (len(ce & me) / len(ce)) if ce else None,
                         "comparator_errors_that_are_also_circuit_errors":
                             (len(ce & me) / len(me)) if me else None})
        s195[b] = {
            "cohen_kappa_per_seed": krows,
            "cohen_kappa_mean": float(np.mean([k for k in krows if k is not None]))
                if any(k is not None for k in krows) else None,
            "error_subsets_per_seed": subs,
            "reading": ("doc 55 SS19.5(a): errors-as-SUBSET reads as 'noisy linear "
                        "model'; DISJOINT errors read as a different inductive bias. "
                        "DESCRIPTIVE, no direction registered."),
        }
        # ---- SS19.5(b) primacy/recency + SS6.9's reported-not-registered strata
        rows_flat = [r for run in g[b] for r in run["heldout_items"]]
        for r in rows_flat:
            it = item_of.get((R.ARM_CFG[b]["part"], r["id"]), {})
            r["_vpair"] = ("%s|%s" % (it.get("verb_slots", {}).get("V_A"),
                                      it.get("verb_slots", {}).get("V_B"))
                           if it.get("verb_slots") else "n/a")
        s69[b] = {
            "by_query_index_PRIMACY_RECENCY": strat(rows_flat,
                                                    lambda r: "q%d" % r["query_index"]),
            "by_surface_order_pair": strat(rows_flat,
                                           lambda r: "/".join(r["order_flags"])),
            "by_surface_order_event1": strat(rows_flat, lambda r: r["order_flags"][0]),
            "by_verb_pair": strat(rows_flat, lambda r: r["_vpair"]),
            "by_queried_verb": strat(rows_flat, lambda r: r["queried_verb"]),
            "val_minus_heldout_per_seed": arms[b]["val_minus_heldout_per_seed"],
            "mean_val_minus_heldout": arms[b]["mean_val_minus_heldout"],
        }

    reported_not_registered = {
        "rule": (
            "doc 55 SS6.9: REPORTED, NOT REGISTERED, and this is deliberate. Per-q "
            "held-out accuracy (the persistence cell q = 1 against the late-binding "
            "cell q = 2), per-surface-order accuracy, the val-heldout selection gap "
            "per seed, and per-verb-pair accuracy. NONE OF THESE MAY BE PROMOTED "
            "ONTO A CRITERION AFTER THE FACT -- Rule 4's second sentence and the "
            "exp44 lesson. doc 55 SS4.2 states the reason: reporting the per-q cells "
            "without a boolean on them pre-empts post-hoc mining of whichever cell "
            "happens to look better, without multiplying criteria."),
        "surface_order_caveat": (
            "exp69's compiled circuit is INVARIANT to the SVO/VSO alternation by "
            "construction -- see exp69_run.py's header and the smoke file's "
            "word_order verdict for the doc-55-internal reconciliation (DERIVATION "
            "DIFFERS / DENOTATION EQUAL, forced by SS4.7, SS5, SS6.1 and SS16.1-Q2). "
            "The per-surface-order numbers are therefore a DATA-COMPOSITION "
            "statistic and are NOT evidence about word-order processing."),
        "per_arm": s69,
    }

    # ======================================================= doc 55 SS8.2 arbiter
    if floor_gate["R2-W"]:
        arbiter_row = "SS8.2 row 1 -- R2-W clears its floor"
        arbiter = ("R2-W CLEARS ITS MATCHED FLOOR (%.4f vs %.4f, %+.4f, %.1f floor-SD "
                   "units). doc 55 SS8.2 row 1: composition converts at the circuit "
                   "level and the reading is SS9.2's, not SS8.2's. The comparator's "
                   "role here is the GUARD only -- it %s R2-W (%.4f vs %.4f, %+.4f) "
                   "-- so no exp69 number may be read as a quantum result (doc 52 "
                   "SS1, doc 55 SS11)."
                   % (arms["R2-W"]["mean"], floors["R2-W"]["mean"],
                      clear["R2-W"]["delta_over_floor"],
                      clear["R2-W"]["sd_units_over_floor_using_floor_sd"],
                      "matches or exceeds" if cmp_ge["R2-W"] else "does NOT match",
                      cmp_arms["R2-W"]["mean"], arms["R2-W"]["mean"],
                      cmp_vs_circuit["R2-W"]))
    elif cmp_clear["R2-W"]["clears_floor"]:
        arbiter_row = "SS8.2 row 2 -- R2-W at floor, comparator clears"
        arbiter = ("CIRCUIT-TRAINABILITY FAILURE AT DEPTH. R2-W sits at its floor "
                   "(%.4f vs %.4f) while the comparator on the IDENTICAL features and "
                   "the IDENTICAL items clears its own floor (%.4f vs %.4f). doc 55 "
                   "SS8.2 row 2: the information is there and is linearly extractable; "
                   "the circuit did not learn it. Observed precedent: doc 53 SS7.1.3 "
                   "on the arity lever. May NOT be reported as 'the information isn't "
                   "there', and may NOT be reported as 'DisCoCirc cannot compose'."
                   % (arms["R2-W"]["mean"], floors["R2-W"]["mean"],
                      cmp_arms["R2-W"]["mean"], cmp_floors["R2-W"]["mean"]))
    else:
        arbiter_row = "SS8.2 row 3 -- R2-W at floor, comparator at floor"
        arbiter = ("INFORMATION / SAMPLE-SIDE. R2-W sits at its floor (%.4f vs %.4f) "
                   "and so does the comparator on the identical features (%.4f vs "
                   "%.4f). doc 55 SS8.2 row 3: the features do not carry the category "
                   "through this task at this sample size, and NO CIRCUIT CLAIM IS "
                   "AVAILABLE EITHER WAY. May not be reported as a circuit result at "
                   "all."
                   % (arms["R2-W"]["mean"], floors["R2-W"]["mean"],
                      cmp_arms["R2-W"]["mean"], cmp_floors["R2-W"]["mean"]))

    # ========================================================= doc 55 SS9.2 rows
    t1_arms_pass = bool(t1_surface_anchor_binds and floor_gate["R1-W"])
    fired = []
    if not anchor_replicates_exp68:
        fired.append({"row": "A0 fails",
                      "reading": "THE RUN STOPS. The exp69 pipeline does not "
                                 "reproduce exp68 on exp68's own data. Hunt the "
                                 "build; interpret nothing.",
                      "may_not_claim": "Nothing, from any arm."})
    if anchor_replicates_exp68 and not t1_surface_anchor_binds:
        fired.append({"row": "R1-A fails its band or its floor",
                      "reading": "THE RUN STOPS -- doc 50 SS5.3's T = 1 anchor "
                                 "discipline. Suspect the generator, the V-suite, or "
                                 "the compilation, in that order.",
                      "may_not_claim": "Nothing, from any arm."})
    if t1_arms_pass and t2_binding_holds:
        fired.append({"row": "T = 1 arms pass, R2-W clears its floor and 0.65",
                      "reading": "BINDING SURVIVES COMPOSITION -- a semantic property "
                                 "of a role-filler is readable through two composed "
                                 "events, with a real category and a real VSO/SVO "
                                 "surface. This is doc 50's reunion, measured.",
                      "may_not_claim": "NOT 'meaning'. NOT quantum advantage. NOT "
                                       "novel-filler generalisation. NOT multi-hop "
                                       "reasoning or discourse resolution."})
    if t1_arms_pass and not t2_binding_holds and cmp_clear["R2-W"]["clears_floor"]:
        fired.append({"row": "T = 1 arms pass, R2-W at floor, comparator clears",
                      "reading": "CIRCUIT-TRAINABILITY FAILURE AT DEPTH. Because the "
                                 "T = 1 arms passed with the identical encoding, "
                                 "features and compilation, the failure is located at "
                                 "COMPOSITION, not at the encoding.",
                      "may_not_claim": "May NOT be reported as 'the information isn't "
                                       "there', nor as 'DisCoCirc cannot compose'."})
    if t1_arms_pass and not t2_binding_holds \
            and not cmp_clear["R2-W"]["clears_floor"]:
        fired.append({"row": "T = 1 arms pass, R2-W at floor, comparator at floor",
                      "reading": "INFORMATION / SAMPLE-SIDE.",
                      "may_not_claim": "May not be reported as a circuit result at all."})
    if floor_gate["R2-W"] and not floor_gate["R2-A"]:
        fired.append({"row": "R2-W clears but R2-A at floor",
                      "reading": "WIDTH CONVERTS WHERE THE ANGLE DOES NOT, AT T = 2 -- "
                                 "the strongest form of the width-as-information "
                                 "claim. Requires SS8.3's comparator clause to agree "
                                 "(it %s)." % ("does" if not s83_blocks_T2
                                               else "does NOT"),
                      "may_not_claim": "May not be generalised past two experiments; "
                                       "the subfamily would have three members, not a "
                                       "literature."})
    if register_beats_angle_at_T2 and not register_beats_angle_at_T1:
        fired.append({"row": "register_beats_angle_at_T2 TRUE while _at_T1 FALSE",
                      "reading": "COMPOSITION IS WHERE THE SECOND FEATURE PAYS. "
                                 "Registered as THE upside surprise; must be "
                                 "confirmed on the seed+1000 draw with all arms "
                                 "present before it is written anywhere.",
                      "may_not_claim": "May not be claimed on the first draw alone "
                                       "(doc 47 SS18)."})
    if not register_beats_angle_at_T2 and not register_beats_angle_at_T1:
        fired.append({"row": "both register booleans FALSE",
                      "reading": "WIDTH-AS-INFORMATION IS A TWO-POINT (THREE-ARM) "
                                 "NEGATIVE, and doc 54 SS17.6's caveat is discharged "
                                 "in the negative direction. doc 54 SS17.4's surviving "
                                 "statement narrows to: 'one RY per qubit converts; a "
                                 "SECOND FEATURE on the second qubit does not add "
                                 "beyond the angle.'",
                      "may_not_claim": "May not be reported as 'width works' in any "
                                       "form (doc 54 SS9.2 row 2, inherited)."})
    if bug_hunt_triggered:
        fired.append({"row": "any arm exceeds its SS12 bound by > 0.06",
                      "reading": "BUG HUNT, NOT RESULT (doc 47 SS4). The bounds are "
                                 "LOWER bounds, so a small excess is expected; a large "
                                 "one means the implementation is not computing what "
                                 "doc 55 says. Arms: %s"
                                 % [b for b in R.BASE_ARMS if bug_hunt[b]],
                      "may_not_claim": "--"})

    # ----------------------------------------------------------------- git note
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                               "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unavailable"

    verdicts = {
        "anchor_replicates_exp68": anchor_replicates_exp68,
        "t1_surface_anchor_binds": t1_surface_anchor_binds,
        "t2_binding_holds": t2_binding_holds,
        "t2_binding_holds_angle": t2_binding_holds_angle,
        "register_beats_angle_at_T2": register_beats_angle_at_T2,
        "register_beats_angle_at_T1": register_beats_angle_at_T1,
        "wazn_binds_T1": wazn_binds_T1, "wazn_binds_T2": wazn_binds_T2,
        "floor_gate_all_arms": all_floors, "floor_gate_per_arm": floor_gate,
        "comparator_matches_or_exceeds_all_arms":
            comparator_matches_or_exceeds_all_arms,
        "comparator_matches_or_exceeds_per_arm": cmp_ge,
        "bug_hunt_triggered": bug_hunt_triggered, "bug_hunt_per_arm": bug_hunt,
        "run_is_interpretable": bool(anchor_replicates_exp68
                                     and t1_surface_anchor_binds and all_floors),
        "criterion_bars": {
            "delta_bar": DELTA_BAR, "floor_sd_multiple": FLOOR_SD,
            "anchor_absolute": ANCHOR_ABS,
            "anchor_band": ANCHOR_BAND, "anchor_point": ANCHOR_POINT,
            "tie_band": TIE_BAND, "a0_tolerance": R.A0_TOLERANCE,
            "threshold_provenance": (
                "0.06 ~ 2 x the binomial SD at n_test = 300 (sqrt(0.25/300) = "
                "0.02887), the bar exp61 SS8 / exp67 SS4 / doc 54 SS7 used. "
                "INHERITED, never chosen for doc 55, and the reason exp69's n_test is "
                "deliberately 300 at every T and in both parts. doc 55 SS6.2 spends a "
                "paragraph refusing to quote exp68's 0.0018 SEED SD against R1-A: "
                "that is a different population (new generator, new surface, fresh "
                "cast draw) and quoting it would be doc 47 SS18.5's failure class."),
            "no_criterion_reads_a_bound": (
                "doc 55 SS6 / SS12. doc 54 SS17.7 is the standing reason: the FA "
                "per-twin bound moved 0.9250 -> 0.9625 AFTER exp68 reported and NO "
                "STORED BOOLEAN MOVED, precisely because none of them read a bound."),
        },
        "registered_expectations_doc55_SS9_1": {
            "anchor_replicates_exp68": "TRUE, bit-exact",
            "t1_surface_anchor_binds": "TRUE",
            "t2_binding_holds": "TRUE, but with the LOWEST CONFIDENCE in the doc -- "
                                "no precedent, every SB rung to date is one event",
            "t2_binding_holds_angle": "TRUE",
            "register_beats_angle_at_T2": "FALSE",
            "register_beats_angle_at_T1": "FALSE",
            "comparator_matches_or_exceeds_all_arms": "TRUE",
            "bug_hunt_triggered": "FALSE",
            "wazn_binds_T1": "no direction registered (doc 55 SS18.4)",
            "wazn_binds_T2": "no direction registered (doc 55 SS18.4)",
        },
        "measured": {
            "means": m,
            "R2W_minus_R2A": d_T2, "R1W_minus_R1A": d_T1,
            "comparator_R2W_minus_R2A": cmp_d_T2,
            "comparator_R1W_minus_R1A": cmp_d_T1,
            "doc55_SS8_3_blocks_T2": s83_blocks_T2,
            "doc55_SS8_3_blocks_T1": s83_blocks_T1,
        },
        "floor_readings_disagree_for": readings_disagree,
        "deliberately_not_registered": (
            "doc 55 SS6.10: (a) NO saturates_ceiling / ceiling_violated criterion -- "
            "doc 52 SS2.2 removed the bound's standing as an instrument on this "
            "inventory. (b) NO T = 1 versus T = 2 accuracy DELTA, because R1 and R2 "
            "are different tasks with different label distributions and differencing "
            "them is a population mismatch. The T = 1 arms are a QUALITATIVE control: "
            "if T = 2 fails while T = 1 passes with the same encoding, the same "
            "compilation path and the same features, the failure is COMPOSITIONAL, "
            "not encoding-level -- and that inference is a pass/fail comparison, "
            "never a subtraction."),
    }

    out = {
        "experiment": "exp69 -- the DisCoCat reunion (doc 55)",
        "spec": SPEC, "spec_commit": SPEC_COMMIT,
        "question": ("Does a bound content-role binding survive COMPOSITION? "
                     "exp60-exp68 all measured binding inside ONE event; exp69 reads "
                     "a semantic property of a role-filler through TWO COMPOSED "
                     "EVENTS, with the one-event case as the control (doc 55 SS1)."),
        "aggregated_from": "%s/*.json (tag=%r)" % (R.CKPT_DIR, a.tag),
        "aggregator": os.path.basename(os.path.abspath(__file__)),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "datasets": ds,
        "predictions_file": PREDICTIONS,
        "vsuite_results": VSUITE, "smoke_results": SMOKE,
        "a0_results": A0_FILE,
        "arms_run": R.ARMS, "n_seeds": R.N_SEEDS, "n_runs": n_runs,
        "sd_convention": "sample sd, ddof=1",
        "training": {"lr": R.LR, "batch": R.BATCH, "max_epochs": R.MAX_EPOCHS,
                     "optimiser": "Adam", "rng_base": R.RNG_BASE,
                     "model_selection": "best in-distribution val accuracy",
                     "seeds_vary": ("OPTIMISER INITIALISATION ONLY -- the features "
                                    "are deterministic constants (doc 55 SS4.4)"),
                     "lineage": "the exp60 / exp61 / exp65 / exp68 loop, unchanged"},
        "arms": arms, "floors": floors, "floor_clearance": clear,
        "a0_anchor": A0,
        "comparator": {"source": COMPARATOR, "arms": {b: {k: v for k, v in
                                                          C["arms"][b].items()
                                                          if k != "runs"}
                                                      for b in R.BASE_ARMS},
                       "floors": {b: {k: v for k, v in C["floors"][b].items()
                                      if k != "runs"} for b in R.FLOORED},
                       "floor_clearance": cmp_clear,
                       "comparator_minus_circuit": cmp_vs_circuit,
                       "arbiter_row_doc55_SS8_2": arbiter_row,
                       "arbiter_reading": arbiter,
                       "function_class_qualifier": C["function_class_qualifier"]},
        "bounds_context_only_doc55_SS12": {
            b: {"family": R.BOUND_FAMILY[b], "bound": bounds[b],
                "measured": m[b], "over_bound_by": over[b]}
            for b in R.BASE_ARMS},
        "composition_cost_doc55_SS18_2": composition,
        "predictions_vs_measured_doc55_SS19_2": predictions_report,
        "margin_accuracy_doc55_SS19_3": s193,
        "purity_contrast_doc55_SS19_4": s194,
        "error_overlap_doc55_SS19_5": s195,
        "reported_not_registered_doc55_SS6_9": reported_not_registered,
        "verdicts": verdicts,
        "outcome_rows_doc55_SS9_2": fired,
        "meta_notes": {
            "scope_doc55_SS11": [
                "NOT 'meaning'. One category (plus one morphological contrast), one "
                "task family, 24 nouns, 2 referents, 2 events, simulation. A positive "
                "shows that a semantic property of a role-filler remains readable "
                "through two composed events -- narrow, and worth exactly that.",
                "NOT quantum advantage. doc 52 SS1 proved it for this encoding family; "
                "doc 55 SS8 registers the comparator as a tie-or-win IN ADVANCE. No "
                "exp69 number may appear in a sentence containing 'advantage'.",
                "NOT 'width works'. Whatever the register arms return, the "
                "two-feature-register subfamily has THREE members total. A "
                "three-member subfamily supports a RECORD, not a design rule, and any "
                "claim must be stated with that count attached.",
                "NOT novel-filler generalisation. The split is over UNSEEN PAIRINGS, "
                "not unseen nouns, and V2 is reported un-gated for that reason.",
                "NOT multi-hop reasoning, anaphora, or discourse. Q69 resolves its "
                "query by VERB IDENTITY over disjoint slot inventories. Two events is "
                "not a discourse, and T = 2 is not 'long-range'.",
                "NOT a statement about AraVec: PC1/PC2 of a 300-dim vector over 44 "
                "words explain a small share of its variance.",
                "NOT a statement about DisCoCat's expressive limits. A negative "
                "locates a failure in THIS build, THIS optimiser, THIS inventory.",
                "The bounds bound THIS feature map and THIS inventory. They are not "
                "ceilings for the task and NO criterion reads against them.",
            ],
            "word_order_reconciliation_ORCHESTRATOR_REVIEW": (
                "doc 55 SS7.3 requires SVO and VSO to compile to DIFFERENT circuits. "
                "exp69 discharges it as DERIVATION-DIFFERS / DENOTATION-EQUAL: the "
                "compiled derivation carries the archived reader's Swap for VSO and "
                "not for SVO (so the compiler genuinely consults the surface, which "
                "is the null-instrument failure SS7.3 exists to catch), and dropping "
                "that Swap is measured to yield EXACTLY the role-reversed circuit, so "
                "the Swap is load-bearing. The denotations are equal, which is doc 55 "
                "SS16.1-Q2's USER-CONFIRMED premise and is FORCED by SS4.7 ('the same "
                "six blocks, reused'), SS5's 12/72 conditioning-group arithmetic (no "
                "surface-order factor) and SS6.1 (A0 must reproduce exp68 bit-exactly "
                "on a dataset that already contains both orders). Consequence carried "
                "into the claim: the circuit is invariant to the alternation, so "
                "SS6.9's per-surface-order numbers are a data-composition statistic. "
                "See results_exp69_smoke_validation.json :: word_order. THE "
                "ORCHESTRATOR MAY VETO THIS READING; it is stated in the open rather "
                "than assumed."),
            "part_b_budget_JUDGMENT_CALL": (
                "doc 55 SS18.4 does not fix Part B's cast budget. Part B has 9 x 15 = "
                "135 possible casts and exp69 DRAWS 80 at the registered seed so the "
                "budget, the 25-cast test split and n_test = 300 mirror Part A exactly "
                "and doc 55 SS6's 0.06 bar transfers exactly rather than "
                "approximately. Recorded as a judgment call in the dataset meta."),
            "confirm_draw_obligation_doc55_SS4_5": (
                "doc 55 SS4.5 registers a CONFIRM DRAW at generator seed + 1000 "
                "(= 70000), to be run FOR ANY POSITIVE VERDICT with ALL ARMS PRESENT "
                "on the same fresh draw, and doc 47 SS18 is the standing reminder that "
                "a headline which survived one draw died on the second. doc 55 SS15 "
                "prices it at 4-6 h. It is NOT included in this file. Any positive "
                "reported here is a FIRST-DRAW result and doc 55 SS9.2's upside-"
                "surprise row explicitly may not be claimed on the first draw alone."),
            "T1_is_a_qualitative_control": (
                "doc 55 SS6.10(b): the T = 1 arms exist so that a T = 2 failure is "
                "read as COMPOSITIONAL rather than as an encoding or pipeline defect. "
                "That inference is a PASS/FAIL comparison and never a subtraction."),
        },
        "git_note": {
            "repo_head": head,
            "spec_commit": SPEC_COMMIT,
            "note": ("doc 55 is committed at %s in qnlp_private_docs and is the "
                     "binding registration through SS19.7. This run wrote NO git "
                     "state; the orchestrator reviews and commits." % SPEC_COMMIT),
        },
        "wall_time_sec": time.time() - t0,
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ console
    print("[exp69] %d runs, %d arms x %d seeds" % (n_runs, len(R.ARMS), R.N_SEEDS))
    print("\n  SS6.1 anchor_replicates_exp68 : %s  (max |delta| %.2e, bit-exact=%s)"
          % (anchor_replicates_exp68, A0["max_abs_per_seed_delta"], A0["bit_exact"]))
    print("\n  arm      T  enc  held-out mean +/- sd      floor        clears  cmp")
    for b in R.BASE_ARMS:
        fl = ("%.4f+/-%.4f" % (floors[b]["mean"], floors[b]["sd"])
              if b in floors else "  (exp68's)  ")
        cl = ("%s(%.1fsd)" % (floor_gate[b], clear[b]["sd_units_over_floor_using_floor_sd"])
              if b in clear else "n/a")
        print("  %-7s %d  %-3s  %.4f +/- %.4f   %s  %-14s %.4f"
              % (b, R.ARM_CFG[b]["T"], R.ARM_CFG[b]["enc"], arms[b]["mean"],
                 arms[b]["sd"], fl, cl, cmp_arms[b]["mean"]))
    print("\n  SS6 booleans:")
    for k in ("t1_surface_anchor_binds", "t2_binding_holds", "t2_binding_holds_angle",
              "register_beats_angle_at_T2", "register_beats_angle_at_T1",
              "wazn_binds_T1", "wazn_binds_T2", "floor_gate_all_arms",
              "comparator_matches_or_exceeds_all_arms", "bug_hunt_triggered",
              "run_is_interpretable"):
        print("    %-40s %s" % (k, verdicts[k]))
    print("\n  SS18.2 composition cost: cc_A=%+.4f  cc_W=%+.4f  cc_M=%+.4f  dcc=%+.4f"
          % (cc_A["mean"], cc_W["mean"], cc_M["mean"], dcc["mean"]))
    print("  SS19.3 Spearman (item / cast):")
    for b in R.BASE_ARMS:
        print("    %-7s %s / %s  (%d distinct margins)"
              % (b, "%.4f" % s193[b]["spearman_item_level"]
                 if s193[b]["spearman_item_level"] is not None else "n/a",
                 "%.4f" % s193[b]["spearman_cast_level"]
                 if s193[b]["spearman_cast_level"] is not None else "n/a",
                 s193[b]["n_distinct_margins"]))
    print("\n  SS8.2 arbiter: %s" % arbiter_row)
    print("  SS9.2 rows fired: %s" % [f["row"] for f in fired])
    print("\n[exp69] wrote %s (%.1fs)" % (a.out, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
