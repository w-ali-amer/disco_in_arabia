# -*- coding: utf-8 -*-
"""exp91_aggregate.py -- doc 84 SS4: named outcomes per (learner, draw, n),
per-draw thresholds, the threshold of record, the registered booleans and the
descriptives, into results_exp91.json. Refuses to overwrite; refuses to run
unless the A91 gates passed.

Reused from exp90_aggregate (imported, unchanged): build_cells (seed mean, 95%
t-CI over seeds, df = n-1), outcome (doc 83 SS8 named-outcome rule), near_miss.

Thresholds (doc 84 SS4, frozen):
  n*_draw(learner) = smallest ladder n with CASE_RELIANT at n AND at every
                     larger ladder n; none -> ">72".
                     BUILD RULING (conservative, flagged): if ANY ladder cell of
                     that (learner, draw) is missing or has < 12 seeds, n*_draw =
                     "INCOMPLETE" -- a partial grid never yields ">72" or a number.
  n*(learner)      = max(n*_primary, n*_confirm), with ">72" above every n;
                     INCOMPLETE if either draw is INCOMPLETE.
Booleans (per init, no AND across inits):
  archM_threshold_defined_<init> = n*(M-<init>) <= 72   (null if INCOMPLETE)
  archM_below_cldisco_<init>     = TRUE iff both defined AND n*(M) < n*(CLD);
                                   null (NOT EVALUATED) if either is undefined.
Gate requirement (record run, not --smoke):
  exp91_gates/{a91render,vsuite,a91units*}.json all passed with intact payload
  hashes, AND the FULL anchor set (6 learners x 12 seeds x 2 anchor cells) from
  the grid's own units reproduces exp90 exactly (checked here, in-process).

    python3 exp91_aggregate.py                       # -> results_exp91.json
    python3 exp91_aggregate.py --smoke --ckpt-dir exp91_smoke --out exp91_smoke/results_smoke.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

import exp90_aggregate as a90                                   # noqa: E402
from exp90_common import canonical_sha256, provenance, write_new_json  # noqa: E402
from exp91_datagen import LADDER, DESIGN_DOC_91                 # noqa: E402
from exp91_run import LEARNERS                                  # noqa: E402

DRAWS = ("primary", "confirm")
ANCHORS = [("primary", 12), ("confirm", 5)]
FIELDS = ("conflict_acc", "consistent_acc", "pred_voc")
GT72 = ">72"
INCOMPLETE = "INCOMPLETE"


def _rank(t):
    return math.inf if t == GT72 else t


def per_draw_threshold(cells, arm, draw):
    outs = {}
    for n in LADDER:
        c = cells.get("%s|%s|n%d" % (draw, arm, n))
        outs[n] = None if c is None else c["outcome"]
    if any(o is None or o == "INSUFFICIENT_SEEDS" for o in outs.values()):
        return INCOMPLETE, outs
    for i, n in enumerate(LADDER):
        if all(outs[m] == "CASE_RELIANT" for m in LADDER[i:]):
            return n, outs
    return GT72, outs


def threshold_of_record(tp, tc):
    if INCOMPLETE in (tp, tc):
        return INCOMPLETE
    return max(tp, tc, key=_rank)


def _logistic(logn, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * logn)))


def logistic_fit(xs_n, ys):
    """Two-parameter logistic of conflict_acc on log n (asymptotes 0 and 1),
    least squares on per-seed points; n at 0.8 = exp((logit(0.8) - a) / b)."""
    x = np.log(np.asarray(xs_n, dtype=float))
    y = np.asarray(ys, dtype=float)
    try:
        (a, b), cov = curve_fit(_logistic, x, y, p0=(0.0, 1.0), maxfev=20000)
    except Exception as e:                                      # noqa: BLE001
        return {"fit_ok": False, "error": repr(e)}
    n80 = None
    if b != 0:
        n80 = float(math.exp((math.log(0.8 / 0.2) - a) / b))
    return {"fit_ok": True, "a": float(a), "b": float(b),
            "param_cov": np.asarray(cov).tolist(), "n_at_0.8": n80,
            "n_at_0.8_within_ladder": bool(n80 is not None and LADDER[0] <= n80 <= LADDER[-1]),
            "n_points": int(len(y))}


def regime_change(cells, arm, draw):
    """First ladder n whose seed-CI on conflict_acc lies strictly inside
    (0.2, 0.8) (excludes both 0.2 and 0.8 on the MIXED side)."""
    for n in LADDER:
        c = cells.get("%s|%s|n%d" % (draw, arm, n))
        if c is None:
            continue
        if c["conflict"]["ci_lo"] > 0.2 and c["conflict"]["ci_hi"] < 0.8:
            return n
    return None


def curves(cells, arm, draw):
    out = {}
    for n in LADDER:
        c = cells.get("%s|%s|n%d" % (draw, arm, n))
        if c is None:
            continue
        out["n%d" % n] = {"outcome": c["outcome"],
                          "conflict_mean": c["conflict"]["mean"],
                          "conflict_ci": [c["conflict"]["ci_lo"], c["conflict"]["ci_hi"]],
                          "consistent_mean": c["consistent"]["mean"],
                          "conflict_per_seed": c["conflict"]["per_seed"],
                          "n_seeds": c["n_seeds"], "n_converged": c["n_converged"],
                          "n_conflict_train": c["n_conflict_train"],
                          "near_miss": a90.near_miss(c)}
    return out


def verify_gate(path):
    g = json.load(open(path, encoding="utf-8"))
    rec = g.pop("payload_sha256", None)
    return {"path": path, "passed": bool(g.get("passed")),
            "payload_sha256": rec, "hash_intact": canonical_sha256(g) == rec,
            "scope": g.get("scope")}


def full_anchor_check(units_by_key):
    rows, missing, n_exact = {}, [], 0
    for draw, n in ANCHORS:
        for arm in LEARNERS:
            for s in range(12):
                k = "%s_%s_n%d_s%d" % (draw, arm, n, s)
                u91 = units_by_key.get(k)
                p90 = os.path.join("exp90_ckpt", "%s_%s_knat_s%d.json" % (draw, arm, s))
                if u91 is None:
                    missing.append(k)
                    continue
                u90 = json.load(open(p90, encoding="utf-8"))
                eq = {f: u91[f] == u90[f] for f in FIELDS}
                rows[k] = eq
                n_exact += all(eq.values())
    return {"n_expected": 144, "n_compared": len(rows), "n_exact": n_exact,
            "missing": missing,
            "failures": [k for k, v in rows.items() if not all(v.values())],
            "passed": bool(not missing and n_exact == 144)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="exp91_ckpt")
    ap.add_argument("--gates-dir", default="exp91_gates")
    ap.add_argument("--out", default="results_exp91.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow non-registered epochs / missing gates; never to results_exp91.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("Rule 3: refusing to overwrite %s" % args.out)
    if args.smoke and os.path.basename(args.out) == "results_exp91.json":
        raise SystemExit("smoke output must not be results_exp91.json")
    # ---------------------------------------------------------- units
    units = []
    for p in sorted(glob.glob(os.path.join(args.ckpt_dir, "*.json"))):
        u = json.load(open(p, encoding="utf-8"))
        if "unit" in u and "exp91" in u:
            assert u["n_conflict_train"] == u["exp91"]["n"], p
            units.append(u)
    if not units:
        raise SystemExit("no exp91 unit JSONs in %s" % args.ckpt_dir)
    eps = {u["max_epochs"] for u in units}
    if not args.smoke and eps != {150}:
        raise SystemExit("non-registered max_epochs %s in %s" % (eps, args.ckpt_dir))
    by_key = {u["unit"]: u for u in units}
    # ---------------------------------------------------------- gates
    gates = {}
    for name in ("a91render", "vsuite"):
        p = os.path.join(args.gates_dir, name + ".json")
        gates[name] = verify_gate(p) if os.path.exists(p) else {"path": p, "passed": False,
                                                                 "missing": True}
    au = sorted(glob.glob(os.path.join(args.gates_dir, "a91units*.json")))
    gates["a91units"] = [verify_gate(p) for p in au]
    anchor_full = full_anchor_check(by_key)
    gates_ok = bool(all(gates[g]["passed"] and gates[g].get("hash_intact")
                        for g in ("a91render", "vsuite"))
                    and gates["a91units"]
                    and all(g["passed"] and g["hash_intact"] for g in gates["a91units"])
                    and anchor_full["passed"])
    if not gates_ok and not args.smoke:
        raise SystemExit("A91 gates not passed -- nothing is read (doc 84 SS3).\n%s\n%s"
                         % (json.dumps(gates, indent=1), json.dumps(
                             {k: v for k, v in anchor_full.items() if k != "failures"}
                             | {"failures": anchor_full["failures"][:10]}, indent=1)))
    # ---------------------------------------------------------- cells
    cells = a90.build_cells(units)
    thr = {}
    for arm in LEARNERS:
        tp, op = per_draw_threshold(cells, arm, "primary")
        tc, oc = per_draw_threshold(cells, arm, "confirm")
        thr[arm] = {"n_star_primary": tp, "n_star_confirm": tc,
                    "n_star": threshold_of_record(tp, tc),
                    "outcomes_primary": {"n%d" % k: v for k, v in op.items()},
                    "outcomes_confirm": {"n%d" % k: v for k, v in oc.items()}}

    def defined(arm):
        t = thr[arm]["n_star"]
        if t == INCOMPLETE:
            return None
        return t != GT72

    B = {}
    for init in ("id", "rand"):
        m, c = "M-" + init, "CLD-" + init
        dm, dc = defined(m), defined(c)
        B["archM_threshold_defined_%s" % init] = {
            "value": dm, "n_star_archM": thr[m]["n_star"]}
        if dm is None or dc is None:
            v, note = None, "INCOMPLETE grid"
        elif dm and dc:
            v, note = bool(thr[m]["n_star"] < thr[c]["n_star"]), "both defined"
        else:
            v, note = None, "NOT EVALUATED: a threshold is undefined (>72); descriptive only (doc 84 SS4/SS6)"
        B["archM_below_cldisco_%s" % init] = {
            "value": v, "n_star_archM": thr[m]["n_star"],
            "n_star_cldisco": thr[c]["n_star"], "note": note}
    desc = {"n_star_mlp": thr["CL-MLP"]["n_star"], "n_star_lstm": thr["CL-LSTM"]["n_star"],
            "per_learner": {}}
    for arm in LEARNERS:
        d = {"curves": {dr: curves(cells, arm, dr) for dr in DRAWS},
             "regime_change_point": {dr: regime_change(cells, arm, dr) for dr in DRAWS},
             "logistic": {}}
        for dr in DRAWS:
            xs, ys = [], []
            for n in LADDER:
                c = cells.get("%s|%s|n%d" % (dr, arm, n))
                if c:
                    xs += [n] * c["n_seeds"]
                    ys += c["conflict"]["per_seed"]
            d["logistic"][dr] = logistic_fit(xs, ys) if len(set(xs)) >= 2 else None
        desc["per_learner"][arm] = d
    res = {
        "experiment": "exp91", "design_doc": DESIGN_DOC_91,
        "statistic": "seed mean, two-sided 95% t-interval over seeds (df=n-1); "
                     "exp90 named-outcome rule (doc 83 SS8)",
        "ladder": LADDER, "learners": LEARNERS, "n_units": len(units),
        "gates": gates, "anchor_full_6x12x2": anchor_full, "gates_ok": gates_ok,
        "cells": cells, "thresholds": thr, "booleans_REGISTERED": B,
        "descriptive_NEVER_PROMOTED": desc,
        "rulings": {
            "incomplete": "a (learner, draw) with any missing / <12-seed ladder cell -> INCOMPLETE",
            "max_order": "'>72' ranks above every ladder n",
            "logistic": "per (learner, draw) on per-seed points; asymptotes fixed at 0 and 1",
            "regime_change": "first n with ci_lo > 0.2 and ci_hi < 0.8",
        },
        "smoke": bool(args.smoke),
        "unit_files_sha256": canonical_sha256(sorted(u["payload_sha256"] for u in units)),
        "provenance": provenance(),
    }
    sha = write_new_json(args.out, res)
    print("[exp91_aggregate] %d units, %d cells -> %s (%s)  gates_ok=%s"
          % (len(units), len(cells), args.out, sha[:12], gates_ok))
    for k, c in cells.items():
        print("  %-26s n=%2d conv=%2d CONS=%.3f CONF=%.3f [%.3f,%.3f] %s"
              % (k, c["n_seeds"], c["n_converged"], c["consistent"]["mean"],
                 c["conflict"]["mean"], c["conflict"]["ci_lo"],
                 c["conflict"]["ci_hi"], c["outcome"]))
    for arm, t in thr.items():
        print("  n* %-9s primary=%s confirm=%s record=%s"
              % (arm, t["n_star_primary"], t["n_star_confirm"], t["n_star"]))
    for k, v in B.items():
        print("  %-30s %s" % (k, v["value"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
