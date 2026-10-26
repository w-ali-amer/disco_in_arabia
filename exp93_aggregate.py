# -*- coding: utf-8 -*-
"""exp93_aggregate.py -- SOLE WRITER of results_exp93.json (doc 88 c1caeb4). Written before any unit output was read.

agreement_routes = TRUE iff on BOTH draws: Q-AGR test_heldout seed mean >= 0.85 AND Q-AGR > Q-BLIND (paired one-sided t,
p < 0.05); symmetric near-miss (+-0.02 on primary -> confirm alone decides, strictly).
G-FLOOR: every Q-BLIND unit has 0 twin prediction mismatches and accuracy exactly 0.5 (else BUILD ERROR -> halt).
G-ORACLE: Q-GOLD held-out seed mean >= 0.85 per draw (else INVALID-CEILING). G-CONV: train acc reported.
Degenerate-minimum descriptive: Q-AGR match tables. Twice-evaluation: accuracies from per-item preds; t by scipy + manual.
"""
import json, math, os, statistics
from scipy import stats
import exp93_common as C

CK, OUT = "exp93_ckpt", "results_exp93.json"
ARMS, DRAWS, SEEDS, BAR, NEAR = ["Q-AGR", "Q-BLIND", "Q-GOLD", "SURF-MLP"], ["primary", "confirm"], range(8), 0.85, 0.02


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {}
    for dr in DRAWS:
        for a in ARMS:
            for s in SEEDS:
                p = os.path.join(CK, "%s_%s_s%d.json" % (dr, a, s))
                assert os.path.exists(p), ("missing", p)
                U[(dr, a, s)] = json.load(open(p, encoding="utf-8"))
    twice = True
    acc = {}
    for k, u in U.items():
        for split in ("test_heldout", "test_seen"):
            rows = u["results"][split]["per_item"]
            a2 = sum(r["pred"] == r["label"] for r in rows) / len(rows)
            twice &= abs(a2 - u["results"][split]["acc"]) < 1e-12
            acc[k + (split,)] = a2
    seeds = lambda dr, a, sp="test_heldout": [acc[(dr, a, s, sp)] for s in SEEDS]
    R = {"per_arm": {}, "gates": {}, "criteria": {}, "descriptive": {}}
    for dr in DRAWS:
        for a in ARMS:
            for sp in ("test_heldout", "test_seen"):
                xs = seeds(dr, a, sp)
                R["per_arm"]["%s|%s|%s" % (dr, a, sp)] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs), "seeds": xs}
    blind_bad = [k for k, u in U.items() if k[1] == "Q-BLIND" and
                 (u["results"]["test_heldout"]["twin_pred_mismatch"] != 0 or abs(u["results"]["test_heldout"]["acc"] - 0.5) > 1e-12)]
    R["gates"]["G_FLOOR"] = {"pass": not blind_bad, "bad_units": ["%s_%s_s%d" % k for k in blind_bad],
                             "max_twin_score_diff": max(U[(dr, "Q-BLIND", s)]["results"]["test_heldout"]["twin_max_score_diff"]
                                                        for dr in DRAWS for s in SEEDS)}
    assert not blind_bad, "G-FLOOR BUILD ERROR: %s" % blind_bad
    g_or = {dr: statistics.mean(seeds(dr, "Q-GOLD")) for dr in DRAWS}
    R["gates"]["G_ORACLE"] = {"means": g_or, "pass": all(v >= BAR for v in g_or.values())}
    R["gates"]["G_CONV_train_acc"] = {"%s_%s_s%d" % k: u["results"]["train"]["acc"] for k, u in U.items()}
    per = {}
    for dr in DRAWS:
        xs, ys = seeds(dr, "Q-AGR"), seeds(dr, "Q-BLIND")
        r = stats.ttest_rel(xs, ys, alternative="greater")
        d = [x - y for x, y in zip(xs, ys)]; m = statistics.mean(d); sd = statistics.stdev(d)
        t2 = math.inf if sd == 0 else m / (sd / math.sqrt(len(d)))
        p2 = (0.0 if m > 0 else 1.0) if sd == 0 else float(stats.t.sf(t2, len(d) - 1))
        p1 = float(r.pvalue) if not math.isnan(r.pvalue) else p2
        twice &= abs(p1 - p2) < 1e-9
        mean = statistics.mean(xs)
        per[dr] = {"mean": mean, "p_vs_blind": p1, "p_manual": p2, "near_miss": abs(mean - BAR) <= NEAR,
                   "strict_pass": mean >= BAR and p1 < 0.05}
    if per["primary"]["near_miss"]:
        rule, val = "PRIMARY NEAR-MISS -> confirm decides", per["confirm"]["strict_pass"]
    else:
        rule, val = "both draws", per["primary"]["strict_pass"] and per["confirm"]["strict_pass"]
    R["criteria"]["agreement_routes"] = {"value": val if R["gates"]["G_ORACLE"]["pass"] else "INVALID-CEILING",
                                         "raw_bool": bool(val), "rule": rule, "per_draw": per}
    R["descriptive"]["match_tables"] = {"%s_s%d" % (dr, s): U[(dr, "Q-AGR", s)]["extra"]["match_table"]
                                        for dr in DRAWS for s in SEEDS}
    R["descriptive"]["seen_minus_heldout_QAGR"] = {dr: statistics.mean(seeds(dr, "Q-AGR", "test_seen")) -
                                                   statistics.mean(seeds(dr, "Q-AGR")) for dr in DRAWS}
    R["twice_evaluation"] = {"agree": bool(twice)}
    R["doc"] = "88 (c1caeb4)"; R["provenance"] = C.provenance()
    C.write_new_json(OUT, R)
    print("agreement_routes:", R["criteria"]["agreement_routes"]["value"], rule,
          {d: (round(x["mean"], 4), x["p_vs_blind"]) for d, x in per.items()})
    print("gates:", R["gates"]["G_FLOOR"]["pass"], R["gates"]["G_ORACLE"], "twice:", twice)


if __name__ == "__main__":
    main()
