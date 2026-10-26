# -*- coding: utf-8 -*-
"""exp98_aggregate.py -- SOLE WRITER of results_exp98.json (doc 93 23d4dfb). Written before any unit output was read.
register_survives_learning (both draws): (i) slope(Q-AGR-REG) = acc(ID) - acc(OODL L=3) <= 0.05; (ii) OODL(Q-AGR-REG) -
OODL(Q-AGR-NOREG) >= 0.10, paired one-sided p < 0.05; (iii) OODL(Q-AGR-REG) >= 0.85.  G-CEIL: Q-GOLD-REG >= 0.85 at L=2,3
(both draws) else INVALID-CEILING.  G-FLOOR: Q-BLIND-REG exact.  depth_advantage (secondary): Q-AGR-REG beats EVERY
ID-competent TF/LSTM config on OODL by >= 0.10, p < 0.05, both draws.  Descriptive: control cost, REG contribution per
L, Q vs R, curves.  Twice-evaluation.
"""
import json, math, os, statistics
from scipy import stats
import exp94_common as C
import exp98_run as R

OUT, CK = "results_exp98.json", R.CK
DRAWS, SEEDS = ["e98p", "e98c"], range(8)
CIRC = ["Q-AGR-REG", "R-AGR-REG", "Q-AGR-NOREG", "Q-GOLD-REG", "Q-BLIND-REG"]
CONFIGS = ["%s_d%d_lr%g" % (f, d, lr) for f in R.GRID for d, lr in R.GRID[f]]


def acc(rows, f=lambda r: True):
    rr = [r for r in rows if f(r)]; return sum(r["pred"] == r["label"] for r in rr) / len(rr)


def one_sided(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]; m = statistics.mean(d); sd = statistics.stdev(d)
    if sd == 0:
        p = 0.0 if m > 0 else 1.0; return {"mean_diff": m, "p": p, "p_manual": p}
    t = m / (sd / math.sqrt(len(d)))
    return {"mean_diff": m, "p": float(stats.ttest_rel(xs, ys, alternative="greater").pvalue), "p_manual": float(stats.t.sf(t, len(d) - 1))}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {(dr, a, s): json.load(open(os.path.join(CK, "%s_%s_s%d.json" % (dr, a, s)), encoding="utf-8"))
         for dr in DRAWS for a in CIRC + CONFIGS for s in SEEDS}
    V, twice = {}, True
    for k, u in U.items():
        r = u["results"]
        v = {"id": acc(r["test_id"]["per_item"]), "oodl": acc(r["test_oodl"]["per_item"]),
             "L2": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 2), "L3": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 3),
             "long0": acc(r["test_long0"]["per_item"])}
        twice &= abs(v["id"] - r["test_id"]["acc"]) < 1e-12 and abs(v["oodl"] - r["test_oodl"]["acc"]) < 1e-12
        v["slope"] = v["id"] - v["L3"]; V[k] = v
    S = lambda dr, a, key: [V[(dr, a, s)][key] for s in SEEDS]
    M = lambda dr, a, key: statistics.mean(S(dr, a, key))
    bad = [(dr, s) for dr in DRAWS for s in SEEDS for t in R.TESTS
           if U[(dr, "Q-BLIND-REG", s)]["results"][t]["twin_pred_mismatch"] != 0]
    assert not bad, "G-FLOOR BUILD ERROR %s" % bad[:3]
    ceil = {"%s|%s" % (dr, L): M(dr, "Q-GOLD-REG", L) for dr in DRAWS for L in ("L2", "L3")}
    ceil_ok = all(v >= 0.85 for v in ceil.values())
    per = {}
    for dr in DRAWS:
        reg_vs = one_sided(S(dr, "Q-AGR-REG", "oodl"), S(dr, "Q-AGR-NOREG", "oodl")); twice &= abs(reg_vs["p"] - reg_vs["p_manual"]) < 1e-9
        per[dr] = {"slope_QAGRREG": M(dr, "Q-AGR-REG", "slope"), "i_slope_ok": M(dr, "Q-AGR-REG", "slope") <= 0.05,
                   "reg_minus_noreg": reg_vs, "ii_reg_ok": reg_vs["mean_diff"] >= 0.10 and reg_vs["p"] < 0.05,
                   "oodl_QAGRREG": M(dr, "Q-AGR-REG", "oodl"), "iii_floor_ok": M(dr, "Q-AGR-REG", "oodl") >= 0.85}
        per[dr]["all"] = per[dr]["i_slope_ok"] and per[dr]["ii_reg_ok"] and per[dr]["iii_floor_ok"]
    rsl = all(p["all"] for p in per.values())
    comp = {dr: [c for c in CONFIGS if M(dr, c, "id") >= 0.95] for dr in DRAWS}
    adv = {}
    for dr in DRAWS:
        rows = {c: one_sided(S(dr, "Q-AGR-REG", "oodl"), S(dr, c, "oodl")) for c in comp[dr]}
        adv[dr] = {"vs": rows, "all": bool(rows) and all(r["mean_diff"] >= 0.10 and r["p"] < 0.05 for r in rows.values())}
    Res = {"criteria": {"register_survives_learning": {"value": rsl if ceil_ok else "INVALID-CEILING", "raw": rsl, "per_draw": per},
                        "depth_advantage": {"value": all(a["all"] for a in adv.values()) if ceil_ok else "INVALID-CEILING", "per_draw": adv}},
           "gates": {"G_CEIL": ceil, "G_CEIL_pass": ceil_ok, "G_FLOOR": True, "competent": comp},
           "table": {"%s|%s" % (dr, a): {k: M(dr, a, k) for k in ("id", "L2", "L3", "oodl", "long0", "slope")} for dr in DRAWS for a in CIRC + CONFIGS},
           "descriptive": {"control_cost_ID": {dr: M(dr, "Q-GOLD-REG", "id") - M(dr, "Q-AGR-REG", "id") for dr in DRAWS},
                           "control_cost_OODL": {dr: M(dr, "Q-GOLD-REG", "oodl") - M(dr, "Q-AGR-REG", "oodl") for dr in DRAWS},
                           "Q_minus_R_AGRREG_oodl": {dr: M(dr, "Q-AGR-REG", "oodl") - M(dr, "R-AGR-REG", "oodl") for dr in DRAWS},
                           "morph_tables": {"%s_s%d" % (dr, s): U[(dr, "Q-AGR-REG", s)]["extra"].get("morph_table") for dr in DRAWS for s in SEEDS}},
           "twice_evaluation": {"agree": bool(twice)}, "doc": "93 (23d4dfb)", "provenance": C.provenance()}
    C.write_new_json(OUT, Res)
    print("register_survives_learning:", Res["criteria"]["register_survives_learning"]["value"], json.dumps(per, default=float)[:600])
    print("depth_advantage:", Res["criteria"]["depth_advantage"]["value"], "ceil", ceil, "twice", twice)
    for a in CIRC + ["TF_d64_lr0.001", "LSTM_d128_lr0.001"]:
        print("  %-18s" % a, {dr: {k: round(M(dr, a, k), 3) for k in ("id", "L2", "L3", "long0", "slope")} for dr in DRAWS})


if __name__ == "__main__":
    main()
