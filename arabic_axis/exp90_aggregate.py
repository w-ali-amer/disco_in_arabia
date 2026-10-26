# -*- coding: utf-8 -*-
"""exp90_aggregate.py -- doc 83 SS8: named outcomes + every boolean, with
per-seed distributions, into results_exp90.json. Rule 3: never overwrites a
results file (refuses if --out exists).

Statistic (doc 70 SS3 rule 3, seed-level): per cell, the seed mean of held-out
accuracy with a two-sided 95% t-interval over seeds (df = n - 1). All seeds
are used (non-converged kept), n_converged printed. A cell with < 12 seeds
gets outcome INSUFFICIENT_SEEDS.

Named outcomes (SS8, frozen): CASE_RELIANT: conflict >= 0.80, lower CI > 0.5,
consistent >= 0.80. POSITION_RELIANT: conflict <= 0.20, upper CI < 0.5,
consistent >= 0.80. MIXED: consistent >= 0.80, neither. UNLEARNED:
consistent < 0.80.

Verdicts are decided on the CONFIRM draw alone (SS8 "Confirm", doc 70 rule 2);
the primary draw is reported as the screen. A boolean with no confirm cell is
PENDING_CONFIRM (null).

Reading choices (flagged in the build report): *_reads_case_* and
archM_default_case_k0_* are TRUE iff the cell's outcome is CASE_RELIANT;
Rung A per-init pass = seed-mean held-out >= 0.90 on BOTH item types (SS7);
"No boolean ANDs two inits" -> rungA_archR / rungA_cldisco / archM_admitted
are reported PER INIT, and each Rung-B boolean carries `counts` = its own
init's Rung A pass on the same draw (G6).
"""
import argparse
import glob
import json
import math
import os
import sys

from scipy import stats

from exp90_common import canonical_sha256, provenance, write_new_json

N_TEST_PER_TYPE = 300
MIN_SEEDS = 12
BAR_HI, BAR_LO, BAR_CONS, BAR_RUNGA = 0.80, 0.20, 0.80, 0.90
ITEM = 1.0 / N_TEST_PER_TYPE


def tci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def outcome(cell):
    if cell["n_seeds"] < MIN_SEEDS:
        return "INSUFFICIENT_SEEDS"
    c, s = cell["conflict"], cell["consistent"]
    if s["mean"] < BAR_CONS:
        return "UNLEARNED"
    if c["mean"] >= BAR_HI and c["ci_lo"] > 0.5:
        return "CASE_RELIANT"
    if c["mean"] <= BAR_LO and c["ci_hi"] < 0.5:
        return "POSITION_RELIANT"
    return "MIXED"


def twin_exact(u):
    """T1 at the item level: each CONFLICT item's prediction equals its
    CONSISTENT twin's (twin_of via datagen ids: <base>__CONF <-> <base>__CONS).
    Returns {view: bool} and the integer identity n_correct sum."""
    ids, types = u["test_item_ids"], u["test_item_types"]
    pos = {i: k for k, i in enumerate(ids)}
    out = {}
    for view in ("voc", "unvoc"):
        pr = u["pred_" + view]
        ok, ncs, ncf = True, 0, 0
        for k, i in enumerate(ids):
            if types[k] != "C":
                continue
            j = pos[i.replace("__CONF", "__CONS")]
            ok &= pr[k] == pr[j]
            ncf += pr[k] == u["test_answers"][k]
            ncs += pr[j] == u["test_answers"][j]
        out[view] = {"twin_preds_identical": bool(ok),
                     "n_correct_sum": ncs + ncf,
                     "complementary": bool(ok and ncs + ncf == N_TEST_PER_TYPE)}
    return out


def build_cells(units):
    cells = {}
    for u in units:
        cells.setdefault((u["draw"], u["arm"], u["cond"]), []).append(u)
    out = {}
    for (d, a, c), us in sorted(cells.items()):
        us = sorted(us, key=lambda x: x["seed"])
        seeds = [x["seed"] for x in us]
        assert len(set(seeds)) == len(seeds), (d, a, c, "duplicate seeds")
        cell = {"draw": d, "arm": a, "cond": c, "k": us[0]["k"],
                "n_seeds": len(us), "seeds": seeds,
                "n_converged": sum(x["converged"] for x in us),
                "n_conflict_train": us[0]["n_conflict_train"],
                "max_epochs": sorted({x["max_epochs"] for x in us})}
        for key, fld in (("conflict", "conflict_acc"),
                         ("consistent", "consistent_acc"),
                         ("unvoc_conflict", "unvoc_conflict_acc"),
                         ("unvoc_consistent", "unvoc_consistent_acc"),
                         ("train", "train_acc"), ("val", "val_acc")):
            xs = [x[fld] for x in us]
            m, sd, lo, hi = tci(xs)
            cell[key] = {"mean": m, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                         "per_seed": xs}
        cell["converged_per_seed"] = [x["converged"] for x in us]
        cell["outcome"] = outcome(cell)
        t1 = [twin_exact(x) for x in us]
        cell["t1_unvoc_exact_all_seeds"] = all(t["unvoc"]["complementary"] for t in t1)
        cell["t1_voc_exact_all_seeds"] = all(t["voc"]["complementary"] for t in t1)
        if a in ("M-id", "M-rand", "M-shuf", "R-und", "R-rand", "R-pure", "CLD-id", "CLD-rand"):
            cell["case_params_per_seed"] = [{"init": x["params_init"],
                                             "final": x["params_final"]} for x in us]
        out["%s|%s|%s" % (d, a, c)] = cell
    return out


def rungA_pass(cells, draw, arm):
    c = cells.get("%s|%s|A" % (draw, arm))
    if c is None or c["n_seeds"] < MIN_SEEDS:
        return None
    return bool(c["consistent"]["mean"] >= BAR_RUNGA and
                c["conflict"]["mean"] >= BAR_RUNGA)


def reads_case(cells, draw, arm, cond):
    c = cells.get("%s|%s|%s" % (draw, arm, cond))
    if c is None or c["n_seeds"] < MIN_SEEDS:
        return None, None
    return c["outcome"] == "CASE_RELIANT", c


def near_miss(c):
    """doc 70 rule 2 (symmetric): distance of the conflict mean / CI bound to
    the CASE_RELIANT bars, in test-item units."""
    if c is None:
        return None
    return {"conflict_mean_minus_0.80_items": (c["conflict"]["mean"] - BAR_HI) / ITEM,
            "ci_lo_minus_0.5_items": (c["conflict"]["ci_lo"] - 0.5) / ITEM,
            "consistent_mean_minus_0.80_items": (c["consistent"]["mean"] - BAR_CONS) / ITEM,
            "within_one_item_of_a_bar": bool(min(
                abs(c["conflict"]["mean"] - BAR_HI), abs(c["conflict"]["ci_lo"] - 0.5),
                abs(c["consistent"]["mean"] - BAR_CONS), abs(c["conflict"]["mean"] - BAR_LO),
                abs(c["conflict"]["ci_hi"] - 0.5)) < ITEM)}


def booleans(cells, draw, gates):
    B = {}
    B["anchor_bit_exact"] = gates.get("a0", {}).get("anchor_bit_exact")
    B["rungA_archR"] = {"und": rungA_pass(cells, draw, "R-und"),
                        "rand": rungA_pass(cells, draw, "R-rand")}
    B["rungA_cldisco"] = {"id": rungA_pass(cells, draw, "CLD-id"),
                          "rand": rungA_pass(cells, draw, "CLD-rand")}
    B["archM_admitted"] = {"id": rungA_pass(cells, draw, "M-id"),
                           "rand": rungA_pass(cells, draw, "M-rand")}
    B["rungA_other_learners_descriptive"] = {
        a: rungA_pass(cells, draw, a) for a in ("P", "M-shuf", "CL-MLP", "CL-LSTM")}
    spec = []
    for init in ("id", "rand"):
        spec.append(("archM_default_case_k0_%s" % init, "M-" + init, "k0"))
    for cond in ("knat", "k25"):
        for init in ("id", "rand"):
            spec.append(("archM_reads_case_%s_%s" % (cond, init), "M-" + init, cond))
    for cond in ("knat", "k25"):
        for init in ("und", "rand"):
            spec.append(("archR_reads_case_%s_%s" % (cond, init), "R-" + init, cond))
    for cond in ("k0", "knat", "k25"):
        for init in ("id", "rand"):
            spec.append(("cldisco_reads_case_%s_%s" % (cond, init), "CLD-" + init, cond))
    for cond in ("k0", "knat"):
        spec.append(("mlp_reads_case_%s" % cond, "CL-MLP", cond))
        spec.append(("lstm_reads_case_%s" % cond, "CL-LSTM", cond))
    for name, arm, cond in spec:
        v, c = reads_case(cells, draw, arm, cond)
        B[name] = {"value": v, "arm": arm, "cond": cond,
                   "outcome": None if c is None else c["outcome"],
                   "conflict_mean": None if c is None else c["conflict"]["mean"],
                   "conflict_ci": None if c is None else [c["conflict"]["ci_lo"], c["conflict"]["ci_hi"]],
                   "consistent_mean": None if c is None else c["consistent"]["mean"],
                   "n_seeds": None if c is None else c["n_seeds"],
                   "n_converged": None if c is None else c["n_converged"],
                   "counts": rungA_pass(cells, draw, arm),
                   "near_miss": near_miss(c)}
    # position complementarity (T1): P on vocalised + every arm unvocalised
    pc_cells = [c for k, c in cells.items() if c["draw"] == draw]
    if pc_cells:
        B["position_complementarity_holds"] = bool(
            all(c["t1_unvoc_exact_all_seeds"] for c in pc_cells) and
            all(c["t1_voc_exact_all_seeds"] for c in pc_cells if c["arm"] == "P"))
    else:
        B["position_complementarity_holds"] = None
    # T4
    t4 = gates.get("t4", {}).get("t4_stationary_R")
    rp = [c for c in pc_cells if c["arm"] == "R-pure"]
    drift_zero = None
    if rp:
        drift_zero = all(p["init"] == p["final"] for c in rp
                         for p in c["case_params_per_seed"])
    B["t4_stationary_R"] = {"value": (None if t4 is None else
                                      bool(t4 and (drift_zero is not False))),
                            "gate_step0_zero_grads": t4,
                            "r_pure_params_unchanged_after_training": drift_zero}
    return B


def floors(cells, draw):
    out = {}
    sd_bin = math.sqrt(0.25 / N_TEST_PER_TYPE)
    for cond in ("A", "k0", "knat", "k25"):
        c = cells.get("%s|M-shuf|%s" % (draw, cond))
        if c is None:
            continue
        adm = {t: abs(c[t]["mean"] - 0.5) <= 2 * sd_bin for t in ("conflict", "consistent")}
        out[cond] = {"conflict_mean": c["conflict"]["mean"],
                     "consistent_mean": c["consistent"]["mean"],
                     "within_2_binomial_sd_of_chance": adm,
                     "gate_applies": cond == "A",
                     "note": ("doc 70 B3 floor-admissibility gate applies at the "
                              "balanced dose; at k0 T3 predicts M-shuf == P")}
    return out


def t5_descriptive(cells, draw):
    out = {}
    for arm in ("M-id", "M-rand"):
        c = cells.get("%s|%s|k0" % (draw, arm))
        if not c:
            continue
        dr = []
        for p in c["case_params_per_seed"]:
            a0, a1 = p["init"]["acc"], p["final"]["acc"]
            dr.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(a0, a1))))
        out[arm] = {"acc_param_drift_per_seed": dr, "outcome": c["outcome"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="exp90_ckpt")
    ap.add_argument("--gates-dir", default="exp90_gates")
    ap.add_argument("--out", default="results_exp90.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow non-registered epochs; never to results_exp90.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("Rule 3: refusing to overwrite %s" % args.out)
    units = [json.load(open(p, encoding="utf-8"))
             for p in sorted(glob.glob(os.path.join(args.ckpt_dir, "primary_*.json"))
                             + glob.glob(os.path.join(args.ckpt_dir, "confirm_*.json")))]
    units = [u for u in units if "unit" in u and "max_epochs" in u]
    if not units:
        raise SystemExit("no unit JSONs in %s" % args.ckpt_dir)
    eps = {u["max_epochs"] for u in units}
    if args.smoke:
        if os.path.basename(args.out) == "results_exp90.json":
            raise SystemExit("smoke output must not be results_exp90.json")
    elif eps != {150}:
        raise SystemExit("non-registered max_epochs %s in %s" % (eps, args.ckpt_dir))
    gates = {}
    for g in ("vsuite", "encoder", "a0", "t4", "t5"):
        p = os.path.join(args.gates_dir, g + ".json")
        if os.path.exists(p):
            gates[g] = json.load(open(p, encoding="utf-8"))
    cells = build_cells(units)
    res = {
        "experiment": "exp90",
        "statistic": "seed mean, two-sided 95% t-interval over seeds (df=n-1)",
        "min_seeds_for_outcome": MIN_SEEDS,
        "n_units": len(units),
        "gates": {g: {"passed": v.get("passed"),
                      "payload_sha256": v.get("payload_sha256")} for g, v in gates.items()},
        "cells": cells,
        "booleans_confirm_DECIDING": booleans(
            {k: c for k, c in cells.items() if c["draw"] == "confirm"}, "confirm", gates),
        "booleans_primary_SCREEN": booleans(
            {k: c for k, c in cells.items() if c["draw"] == "primary"}, "primary", gates),
        "floor_admissibility": {d: floors(cells, d) for d in ("primary", "confirm")},
        "t5_acc_drift_descriptive": {d: t5_descriptive(cells, d) for d in ("primary", "confirm")},
        "section9_note": ("SS9 cell read at knat: row = Arch-M if admitted else "
                          "Arch-R; column = CL-DisCoCat. Per-init outcomes are "
                          "in the booleans; the cell is assigned by the "
                          "orchestrator. No SS9 sentence may leave the private "
                          "repo before the SS12 judge verdicts are recorded."),
        "smoke": bool(args.smoke),
        "unit_files_sha256": canonical_sha256(sorted(u["payload_sha256"] for u in units)),
        "provenance": provenance(),
    }
    sha = write_new_json(args.out, res)
    print("[exp90_aggregate] %d units, %d cells -> %s (%s)"
          % (len(units), len(cells), args.out, sha[:12]))
    for k, c in cells.items():
        print("  %-28s n=%2d conv=%2d CONS=%.3f CONF=%.3f [%.3f,%.3f] %s"
              % (k, c["n_seeds"], c["n_converged"], c["consistent"]["mean"],
                 c["conflict"]["mean"], c["conflict"]["ci_lo"],
                 c["conflict"]["ci_hi"], c["outcome"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
