# -*- coding: utf-8 -*-
"""exp97_aggregate.py -- SOLE WRITER of results_exp97.json (doc 92 24c4f39 + §4 72727a4). Written before any unit output.
Fresh draw only (final for this question). slope = acc(ID) - acc(OODL, L=3); normalised = slope / (acc(ID) - 0.5).
field_helps_depth: slope(R-INV) - slope(Q-INV) >= 0.05, paired one-sided p < 0.05.
room_explains_field: d = slope(R-8) - slope(Q-INV): CLOSES if two-sided 95% CI of d inside [-0.05, 0.05];
  DOES NOT CLOSE if mean d >= 0.05 and one-sided p < 0.05; else INCONCLUSIVE.
Descriptive: ID/L2/L3/LONG0/slope/normalised slope/train acc/best epoch/params per arm; R-8-LOCAL; informative case
(R-8 ID within 0.02 of Q-INV ID but slope within 0.05 of R-INV's). Twice-evaluation.
"""
import json, math, os, statistics
from scipy import stats
import exp94_common as C

OUT, CK = "results_exp97.json", "exp97_ckpt"
ARMS = ["Q-INV", "R-INV", "R-8", "R-8-LOCAL", "R-UNC-GRAM"]
SEEDS, DR = range(8), "fresh"


def acc(rows, f=lambda r: True):
    rr = [r for r in rows if f(r)]
    return sum(r["pred"] == r["label"] for r in rr) / len(rr)


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {(a, s): json.load(open(os.path.join(CK, "%s_%s_s%d.json" % (DR, a, s)), encoding="utf-8")) for a in ARMS for s in SEEDS}
    V, twice = {}, True
    for k, u in U.items():
        r = u["results"]
        v = {"id": acc(r["test_id"]["per_item"]), "L2": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 2),
             "L3": acc(r["test_oodl"]["per_item"], lambda x: x["L"] == 3), "long0": acc(r["test_long0"]["per_item"]),
             "train": r["train"]["acc"], "best_epoch": u["best_epoch"], "params": u["n_params"]}
        twice &= abs(v["id"] - r["test_id"]["acc"]) < 1e-12
        v["slope"] = v["id"] - v["L3"]; v["norm_slope"] = v["slope"] / max(v["id"] - 0.5, 1e-9)
        V[k] = v
    S = lambda a, key: [V[(a, s)][key] for s in SEEDS]
    table = {a: {key: statistics.mean(S(a, key)) for key in ("id", "L2", "L3", "long0", "slope", "norm_slope", "train", "best_epoch", "params")}
             for a in ARMS}
    d1 = [x - y for x, y in zip(S("R-INV", "slope"), S("Q-INV", "slope"))]
    m1, sd1 = statistics.mean(d1), statistics.stdev(d1)
    p1 = float(stats.ttest_rel(S("R-INV", "slope"), S("Q-INV", "slope"), alternative="greater").pvalue) if sd1 else (0.0 if m1 > 0 else 1.0)
    p1m = float(stats.t.sf(m1 / (sd1 / math.sqrt(8)), 7)) if sd1 else p1
    twice &= abs(p1 - p1m) < 1e-9
    fhd = {"mean_diff": m1, "p": p1, "value": bool(m1 >= 0.05 and p1 < 0.05)}
    d2 = [x - y for x, y in zip(S("R-8", "slope"), S("Q-INV", "slope"))]
    m2, sd2 = statistics.mean(d2), statistics.stdev(d2)
    h = stats.t.ppf(0.975, 7) * sd2 / math.sqrt(8)
    ci = [m2 - h, m2 + h]
    p2 = float(stats.ttest_1samp(d2, 0.0, alternative="greater").pvalue) if sd2 else (0.0 if m2 > 0 else 1.0)
    if ci[0] >= -0.05 and ci[1] <= 0.05:
        outcome = "CLOSES"
    elif m2 >= 0.05 and p2 < 0.05:
        outcome = "DOES NOT CLOSE"
    else:
        outcome = "INCONCLUSIVE"
    ref = {"mean_diff": m2, "ci95": ci, "p_one_sided": p2, "value": outcome}
    informative = (abs(table["R-8"]["id"] - table["Q-INV"]["id"]) <= 0.02 and abs(table["R-8"]["slope"] - table["R-INV"]["slope"]) <= 0.05)
    Res = {"table": table, "criteria": {"field_helps_depth": fhd, "room_explains_field": ref},
           "descriptive": {"informative_case_R8_QINV_level_RINV_slope": informative,
                           "R8LOCAL_minus_QINV_slope": statistics.mean(x - y for x, y in zip(S("R-8-LOCAL", "slope"), S("Q-INV", "slope"))),
                           "G_REAL_max_imag": max(U[k]["extra"].get("G_REAL_max_imag", 0.0) for k in U)},
           "twice_evaluation": {"agree": bool(twice)}, "doc": "92 (24c4f39 + 72727a4)", "provenance": C.provenance()}
    C.write_new_json(OUT, Res)
    print("field_helps_depth:", fhd); print("room_explains_field:", ref)
    for a, v in table.items():
        print("  %-11s" % a, {k: round(x, 3) for k, x in v.items()})
    print("informative:", informative, "twice:", twice)


if __name__ == "__main__":
    main()
