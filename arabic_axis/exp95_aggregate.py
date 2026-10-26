# -*- coding: utf-8 -*-
"""exp95_aggregate.py -- SOLE WRITER of results_exp95.json (doc 90 1217f96). Written before any unit output.

depth_generalises: Q-AGR TEST-OODL seed mean >= 0.85 on both draws, conditional on G-CEIL (Q-GOLD >= 0.85 at L=2 and L=3,
  both draws; else INVALID-CEILING).  depth_advantage: Q-AGR beats EVERY ID-competent config (TEST-ID >= 0.95) in EVERY
  family on TEST-OODL by >= 0.10, paired one-sided p < 0.05, both draws; symmetric near-miss (+-0.02, confirm decides).
Curves: per-L accuracy for all arms; Q-AGR - Q-GOLD per L; Q-AGR(L) - Q-AGR(ID); LONG0 vs OODL.  Twice-evaluation.
"""
import json, math, os, statistics
from scipy import stats
import exp94_common as C
import exp95_run as R

OUT, CK = "results_exp95.json", R.CK
DRAWS, SEEDS, BAR, MARGIN, NEAR = ["primary", "confirm"], range(8), 0.85, 0.10, 0.02
CIRC = ["Q-AGR", "CL-AGR", "Q-BLIND", "Q-GOLD"]
CONFIGS = ["%s_d%d_lr%g" % (f, d, lr) for f in R.GRID for d, lr in R.GRID[f]]


def paired(xs, ys, alt="greater"):
    d = [x - y for x, y in zip(xs, ys)]; m = statistics.mean(d); sd = statistics.stdev(d)
    if sd == 0:
        p = (0.0 if m > 0 else 1.0) if alt == "greater" else (0.0 if m else 1.0)
        return {"mean_diff": m, "p": p, "p_manual": p}
    t = m / (sd / math.sqrt(len(d)))
    p1 = float(stats.ttest_rel(xs, ys, alternative=alt).pvalue)
    p2 = float(stats.t.sf(t, len(d) - 1)) if alt == "greater" else float(2 * stats.t.sf(abs(t), len(d) - 1))
    return {"mean_diff": m, "p": p1, "p_manual": p2}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {}
    for dr in DRAWS:
        for a in CIRC + CONFIGS:
            for s in SEEDS:
                p = os.path.join(CK, "%s_%s_s%d.json" % (dr, a, s)); assert os.path.exists(p), p
                U[(dr, a, s)] = json.load(open(p, encoding="utf-8"))
    A, twice = {}, True
    for k, u in U.items():
        for t in R.TESTS:
            rows = u["results"][t]["per_item"]
            a1 = sum(r["pred"] == r["label"] for r in rows) / len(rows); twice &= abs(a1 - u["results"][t]["acc"]) < 1e-12
            A[k + (t,)] = a1
            for Lv in sorted({r["L"] for r in rows}):
                rr = [r for r in rows if r["L"] == Lv]
                A[k + ("%s_L%d" % (t, Lv),)] = sum(r["pred"] == r["label"] for r in rr) / len(rr)
    S = lambda dr, a, t: [A[(dr, a, s, t)] for s in SEEDS]
    keys = sorted({k[3] for k in A})
    Res = {"per_arm": {"%s|%s|%s" % (dr, a, t): statistics.mean(S(dr, a, t)) for dr in DRAWS for a in CIRC + CONFIGS
                       for t in keys if (dr, a, 0, t) in A}, "gates": {}, "criteria": {}, "curves": {}}
    bad = [(dr, s, t) for dr in DRAWS for s in SEEDS for t in R.TESTS
           if U[(dr, "Q-BLIND", s)]["results"][t]["twin_pred_mismatch"] != 0 or abs(A[(dr, "Q-BLIND", s, t)] - 0.5) > 1e-12]
    Res["gates"]["G_FLOOR"] = {"pass": not bad}; assert not bad, "G-FLOOR BUILD ERROR"
    ceil = {"%s|L%d" % (dr, L): statistics.mean(S(dr, "Q-GOLD", "test_oodl_L%d" % L)) for dr in DRAWS for L in (2, 3)}
    ceil_ok = all(v >= BAR for v in ceil.values())
    Res["gates"]["G_CEIL"] = {"per_L": ceil, "pass": ceil_ok,
                              "id": {dr: statistics.mean(S(dr, "Q-GOLD", "test_id")) for dr in DRAWS}}
    comp = {dr: [c for c in CONFIGS if statistics.mean(S(dr, c, "test_id")) >= 0.95] for dr in DRAWS}
    Res["gates"]["G_BASE_ID_competent"] = comp
    gen = {dr: statistics.mean(S(dr, "Q-AGR", "test_oodl")) for dr in DRAWS}
    Res["criteria"]["depth_generalises"] = {"value": all(v >= BAR for v in gen.values()) if ceil_ok else "INVALID-CEILING",
                                            "means": gen}
    per = {}
    for dr in DRAWS:
        rows = {}
        for c in comp[dr]:
            pt = paired(S(dr, "Q-AGR", "test_oodl"), S(dr, c, "test_oodl")); twice &= abs(pt["p"] - pt["p_manual"]) < 1e-9
            rows[c] = dict(pt, strict=pt["mean_diff"] >= MARGIN and pt["p"] < 0.05, near=abs(pt["mean_diff"] - MARGIN) <= NEAR)
        per[dr] = {"vs_config": rows, "all_strict": bool(rows) and all(r["strict"] for r in rows.values()),
                   "any_near": any(r["near"] for r in rows.values())}
    if per["primary"]["any_near"] and not per["primary"]["all_strict"]:
        rule, val = "PRIMARY NEAR-MISS -> confirm decides", per["confirm"]["all_strict"]
    else:
        rule, val = "both draws", per["primary"]["all_strict"] and per["confirm"]["all_strict"]
    Res["criteria"]["depth_advantage"] = {"value": val if ceil_ok else "INVALID-CEILING", "rule": rule, "per_draw": per}
    for dr in DRAWS:
        cur = {}
        for a in CIRC + CONFIGS:
            cur[a] = {k: statistics.mean(S(dr, a, k)) for k in keys if (dr, a, 0, k) in A and ("_L" in k or k in R.TESTS)}
        Res["curves"][dr] = cur
        Res["curves"][dr + "|QAGR_minus_QGOLD_per_L"] = {k: cur["Q-AGR"][k] - cur["Q-GOLD"][k] for k in cur["Q-AGR"] if "_L" in k}
        Res["curves"][dr + "|QAGR_L_minus_ID"] = {k: cur["Q-AGR"][k] - cur["Q-AGR"]["test_id"] for k in cur["Q-AGR"] if "_L" in k}
    Res["twice_evaluation"] = {"agree": bool(twice)}
    Res["doc"] = "90 (1217f96)"; Res["provenance"] = C.provenance()
    C.write_new_json(OUT, Res)
    print("depth_generalises:", Res["criteria"]["depth_generalises"], "\ndepth_advantage:", val, rule, "\nceil:", ceil, "twice:", twice)


if __name__ == "__main__":
    main()
