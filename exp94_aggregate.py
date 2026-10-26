# -*- coding: utf-8 -*-
"""exp94_aggregate.py -- SOLE WRITER of results_exp94.json (doc 89 7fb995a + §12 e6888f1). Written before any unit output.

structure_advantage_X (X in C, K) = TRUE iff Q-AGR beats EVERY ID-competent config (TEST-ID >= 0.95 on the seed mean)
  in EVERY baseline family by >= 0.10 with paired one-sided t p < 0.05 on BOTH draws (§12.4); symmetric near-miss on the
  primary margin (+-0.02 -> confirm alone decides). structure_generalises_C: Q-AGR OODC seed mean >= 0.85 both draws.
G-FLOOR: Q-BLIND exactly 0.5 with 0 twin mismatches on every test set.  G-CEIL: Q-GOLD >= 0.85 per condition per draw.
G-BASE-EXPR: TF-FULL OODC >= 0.95. quantum_vs_twin: Q-AGR - CL-AGR on OODC, two-sided paired t (edge only if >= 0.05,
p < 0.05, both draws). Per-ptype OODC accuracy. Twice-evaluation: accuracies from per-item rows; t by scipy and by hand.
"""
import json, math, os, statistics
from scipy import stats
import exp94_common as C
import exp94_run as R

OUT = "results_exp94.json"; CK = R.CK
DRAWS, SEEDS, BAR, MARGIN, NEAR = ["primary", "confirm"], range(8), 0.85, 0.10, 0.02
CIRC = ["Q-AGR", "CL-AGR", "Q-BLIND", "Q-GOLD"]
CONFIGS = ["%s_d%d_lr%g" % (f, d, lr) for f in R.GRID for d, lr in R.GRID[f]]
TESTS = R.TESTS


def load(dr, arm, s):
    p = os.path.join(CK, "%s_%s_s%d.json" % (dr, arm, s))
    assert os.path.exists(p), ("missing", p)
    return json.load(open(p, encoding="utf-8"))


def acc_rows(rows, ptype=None):
    rows = [r for r in rows if ptype is None or r["ptype"] == ptype]
    return sum(r["pred"] == r["label"] for r in rows) / len(rows)


def paired(xs, ys, alternative="greater"):
    d = [x - y for x, y in zip(xs, ys)]; m = statistics.mean(d); sd = statistics.stdev(d)
    if sd == 0:
        p = (0.0 if m > 0 else 1.0) if alternative == "greater" else (0.0 if m != 0 else 1.0)
        return {"mean_diff": m, "t": math.copysign(math.inf, m) if m else 0.0, "p": p, "p_manual": p}
    t = m / (sd / math.sqrt(len(d)))
    p1 = float(stats.ttest_rel(xs, ys, alternative=alternative).pvalue)
    p2 = float(stats.t.sf(t, len(d) - 1)) if alternative == "greater" else float(2 * stats.t.sf(abs(t), len(d) - 1))
    return {"mean_diff": m, "t": t, "p": p1, "p_manual": p2}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {(dr, a, s): load(dr, a, s) for dr in DRAWS for a in CIRC + CONFIGS + ["TF-FULL"] for s in SEEDS}
    A, twice = {}, True
    for k, u in U.items():
        for t in TESTS:
            rows = u["results"][t]["per_item"]
            a1 = acc_rows(rows); twice &= abs(a1 - u["results"][t]["acc"]) < 1e-12
            A[k + (t,)] = a1
            if t == "test_oodc":
                A[k + ("oodc_T1",)] = acc_rows(rows, "T1"); A[k + ("oodc_T2",)] = acc_rows(rows, "T2")
    S = lambda dr, a, t: [A[(dr, a, s, t)] for s in SEEDS]
    Res = {"per_arm": {}, "gates": {}, "criteria": {}, "descriptive": {}}
    for dr in DRAWS:
        for a in CIRC + CONFIGS + ["TF-FULL"]:
            for t in TESTS + ["oodc_T1", "oodc_T2"]:
                xs = S(dr, a, t)
                Res["per_arm"]["%s|%s|%s" % (dr, a, t)] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs)}
    # gates
    blind_bad = [(dr, s, t) for dr in DRAWS for s in SEEDS for t in TESTS
                 if U[(dr, "Q-BLIND", s)]["results"][t]["twin_pred_mismatch"] != 0 or abs(A[(dr, "Q-BLIND", s, t)] - 0.5) > 1e-12]
    Res["gates"]["G_FLOOR"] = {"pass": not blind_bad, "bad": blind_bad[:10]}
    assert not blind_bad, "G-FLOOR BUILD ERROR"
    Res["gates"]["G_CEIL"] = {"%s|%s" % (dr, t): statistics.mean(S(dr, "Q-GOLD", t)) for dr in DRAWS for t in TESTS}
    ceil_ok = {t: all(Res["gates"]["G_CEIL"]["%s|%s" % (dr, t)] >= BAR for dr in DRAWS) for t in TESTS}
    Res["gates"]["G_CEIL_pass"] = ceil_ok
    competent = {dr: [c for c in CONFIGS if statistics.mean(S(dr, c, "test_id")) >= 0.95] for dr in DRAWS}
    Res["gates"]["G_BASE_ID_competent_configs"] = competent
    Res["gates"]["G_BASE_EXPR"] = {dr: statistics.mean(S(dr, "TF-FULL", "test_oodc")) for dr in DRAWS}
    # criteria
    def advantage(t):
        per = {}
        for dr in DRAWS:
            rows = {}
            for c in competent[dr]:
                pt = paired(S(dr, "Q-AGR", t), S(dr, c, t))
                twice_ok = abs(pt["p"] - pt["p_manual"]) < 1e-9
                rows[c] = dict(pt, margin_ok=pt["mean_diff"] >= MARGIN, near=abs(pt["mean_diff"] - MARGIN) <= NEAR,
                               strict=(pt["mean_diff"] >= MARGIN and pt["p"] < 0.05), twice=twice_ok)
            per[dr] = {"vs_config": rows, "all_strict": bool(rows) and all(r["strict"] for r in rows.values()),
                       "any_near": any(r["near"] for r in rows.values()), "n_competent": len(rows)}
        if per["primary"]["any_near"] and not per["primary"]["all_strict"]:
            rule, val = "PRIMARY NEAR-MISS -> confirm decides", per["confirm"]["all_strict"]
        else:
            rule, val = "both draws", per["primary"]["all_strict"] and per["confirm"]["all_strict"]
        if not ceil_ok[t]:
            val = "INVALID-CEILING"
        return {"value": val, "rule": rule, "per_draw": per}
    Res["criteria"]["structure_advantage_C"] = advantage("test_oodc")
    Res["criteria"]["structure_advantage_K"] = advantage("test_oodk")
    gen = {dr: statistics.mean(S(dr, "Q-AGR", "test_oodc")) for dr in DRAWS}
    Res["criteria"]["structure_generalises_C"] = {"value": all(v >= BAR for v in gen.values()) if ceil_ok["test_oodc"]
                                                  else "INVALID-CEILING", "means": gen}
    qvt = {dr: paired(S(dr, "Q-AGR", "test_oodc"), S(dr, "CL-AGR", "test_oodc"), "two-sided") for dr in DRAWS}
    Res["criteria"]["quantum_vs_twin"] = {"per_draw": qvt, "quantum_edge": all(v["mean_diff"] >= 0.05 and v["p"] < 0.05
                                                                                  for v in qvt.values())}
    for c in Res["criteria"].values():
        for dr in c.get("per_draw", {}).values():
            if isinstance(dr, dict) and "vs_config" in dr:
                twice &= all(r["twice"] for r in dr["vs_config"].values())
    # descriptive: practitioner's pick, orientation, params
    Res["descriptive"]["practitioner_pick"] = {"%s|%s|s%d" % (dr, f, s): R.select(dr, f, s) for dr in DRAWS for f in R.GRID for s in SEEDS}
    Res["descriptive"]["morph_tables_QAGR"] = {"%s_s%d" % (dr, s): U[(dr, "Q-AGR", s)]["extra"]["morph_table"] for dr in DRAWS for s in SEEDS}
    Res["descriptive"]["n_params"] = {a: U[("primary", a, 0)]["n_params"] for a in CIRC + CONFIGS}
    Res["twice_evaluation"] = {"agree": bool(twice)}
    Res["doc"] = "89 (7fb995a) + §12 (e6888f1)"; Res["provenance"] = C.provenance()
    C.write_new_json(OUT, Res)
    for k, v in Res["criteria"].items():
        print(k, v.get("value", v.get("quantum_edge")))
    print("competent:", {dr: len(v) for dr, v in competent.items()}, "ceil:", ceil_ok, "twice:", twice)


if __name__ == "__main__":
    main()
