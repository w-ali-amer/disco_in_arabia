# -*- coding: utf-8 -*-
"""exp96_aggregate.py -- SOLE WRITER of results_exp96.json (doc 91 8929d26). Written before any unit output was read.
slope = acc(test_id) - acc(test_oodl, L=3 items), per seed.
field_costs_depth: slope(Q-INV) - slope(R-INV) >= 0.05, paired one-sided p < 0.05, both exp95 draws.
readout_costs_depth: slope(Q-INV) - slope(Q-FREE) >= 0.05, same. Symmetric near-miss (+-0.02; confirm decides).
Descriptive: 2x2 table, interaction, R-UNC-GRAM + exp95 Q-GOLD reference rows, LONG0, G-REAL. Twice-evaluation.
"""
import json, math, os, statistics
from scipy import stats
import exp94_common as C

OUT, CK = "results_exp96.json", "exp96_ckpt"
ARMS = ["Q-INV", "R-INV", "Q-FREE", "R-FREE", "R-UNC-GRAM"]
DRAWS, SEEDS, MARGIN, NEAR = ["primary", "confirm"], range(8), 0.05, 0.02


def acc(rows, pred=lambda r: True):
    rr = [r for r in rows if pred(r)]
    return sum(r["pred"] == r["label"] for r in rr) / len(rr)


def paired(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]; m = statistics.mean(d); sd = statistics.stdev(d)
    if sd == 0:
        p = 0.0 if m > 0 else 1.0; return {"mean_diff": m, "p": p, "p_manual": p}
    t = m / (sd / math.sqrt(len(d)))
    return {"mean_diff": m, "p": float(stats.ttest_rel(xs, ys, alternative="greater").pvalue),
            "p_manual": float(stats.t.sf(t, len(d) - 1))}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {(dr, a, s): json.load(open(os.path.join(CK, "%s_%s_s%d.json" % (dr, a, s)), encoding="utf-8"))
         for dr in DRAWS for a in ARMS for s in SEEDS}
    M, twice = {}, True
    for k, u in U.items():
        r = u["results"]
        v = {"id": acc(r["test_id"]["per_item"]), "L2": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 2),
             "L3": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 3), "long0": acc(r["test_long0"]["per_item"])}
        twice &= abs(v["id"] - r["test_id"]["acc"]) < 1e-12
        v["slope"] = v["id"] - v["L3"]; M[k] = v
    S = lambda dr, a, key: [M[(dr, a, s)][key] for s in SEEDS]
    Res = {"table": {"%s|%s" % (dr, a): {key: statistics.mean(S(dr, a, key)) for key in ("id", "L2", "L3", "long0", "slope")}
                     for dr in DRAWS for a in ARMS}, "criteria": {}}
    def crit(a, b):
        per = {}
        for dr in DRAWS:
            pt = paired(S(dr, a, "slope"), S(dr, b, "slope"))
            per[dr] = dict(pt, strict=pt["mean_diff"] >= MARGIN and pt["p"] < 0.05, near=abs(pt["mean_diff"] - MARGIN) <= NEAR)
            nonlocal_twice.append(abs(pt["p"] - pt["p_manual"]) < 1e-9)
        if per["primary"]["near"] and not per["primary"]["strict"]:
            rule, val = "PRIMARY NEAR-MISS -> confirm decides", per["confirm"]["strict"]
        else:
            rule, val = "both draws", per["primary"]["strict"] and per["confirm"]["strict"]
        return {"value": bool(val), "rule": rule, "per_draw": per}
    nonlocal_twice = []
    Res["criteria"]["field_costs_depth"] = crit("Q-INV", "R-INV")
    Res["criteria"]["readout_costs_depth"] = crit("Q-INV", "Q-FREE")
    Res["descriptive"] = {
        "interaction_field_x_readout": {dr: statistics.mean([(M[(dr, "Q-INV", s)]["slope"] - M[(dr, "R-INV", s)]["slope"]) -
                                                             (M[(dr, "Q-FREE", s)]["slope"] - M[(dr, "R-FREE", s)]["slope"])
                                                             for s in SEEDS]) for dr in DRAWS},
        "field_under_FREE": {dr: paired(S(dr, "Q-FREE", "slope"), S(dr, "R-FREE", "slope")) for dr in DRAWS},
        "readout_under_REAL": {dr: paired(S(dr, "R-INV", "slope"), S(dr, "R-FREE", "slope")) for dr in DRAWS},
        "G_REAL_max_imag": max(U[k]["extra"].get("G_REAL_max_imag", 0.0) for k in U if k[1].startswith("R-") and k[1] != "R-UNC-GRAM"),
        "exp95_QGOLD_reference": "results_exp95.json curves (Circuit4, not re-run)"}
    twice &= all(nonlocal_twice)
    Res["twice_evaluation"] = {"agree": bool(twice)}
    Res["stage2_required"] = any(c["value"] for c in Res["criteria"].values())
    Res["doc"] = "91 (8929d26)"; Res["provenance"] = C.provenance()
    C.write_new_json(OUT, Res)
    for k, v in Res["criteria"].items():
        print(k, v["value"], {d: (round(x["mean_diff"], 4), round(x["p"], 4)) for d, x in v["per_draw"].items()})
    for k, v in Res["table"].items():
        print("  %-20s" % k, {kk: round(vv, 3) for kk, vv in v.items()})
    print("twice", twice, "stage2", Res["stage2_required"], "G-REAL", Res["descriptive"]["G_REAL_max_imag"])


if __name__ == "__main__":
    main()
