# -*- coding: utf-8 -*-
"""exp92_aggregate.py -- SOLE WRITER of results_exp92.json (doc 87 59f6621, §12 54ddb04). Written before any unit
output was read.

Frozen criteria (doc 87 §7, §6, §12.3):
  anti_transfers / caus_transfers: Q-COMP held-out (VII / IV) seed mean >= 0.85 AND Q-COMP > Q-CELL, paired one-sided
    t over 8 seeds, p < 0.05 -- on BOTH draws. Symmetric near-miss: a primary seed mean within 0.02 of 0.85 is
    NEAR-MISS and the confirm draw alone decides (strictly).
  G-ORACLE: OP-ORACLE held-out seed mean >= 0.85 on each draw, else that alternation is INVALID-CEILING.
  G-FLOOR (amended §12.3b): two-sided 95% t-CI of the 8 Q-CELL seed means includes 0.5 on each draw, else INVALID.
  G-CONV: units with Form I train accuracy < 0.90 are reported; none dropped.
Q-COMP-SG is DIAGNOSTIC: the same bar is evaluated and reported, never entering the verdict.
Twice-evaluation: seed accuracies are recomputed from per-item predictions and the paired t by a second implementation.
"""
import glob, json, math, os, statistics
from scipy import stats
import exp92_common as C

CK = "exp92_ckpt"
OUT = "results_exp92.json"
ARMS = ["Q-COMP", "Q-CELL", "OP-ORACLE", "Q-COMP-SG", "CL-DISCO", "CL-MLP-FACT"]
DRAWS = ["primary", "confirm"]
ALTS = {"ANTI": "VII", "CAUS": "IV"}
BAR, NEAR, SEEDS = 0.85, 0.02, range(8)


def load(draw, arm, seed, n_i=None):
    p = os.path.join(CK, "%s_%s%s_s%d.json" % (draw, arm, "_nI%d" % n_i if n_i else "", seed))
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired_t(xs, ys):
    """path 1 (scipy): H_a mean(xs) > mean(ys)."""
    r = stats.ttest_rel(xs, ys, alternative="greater")
    return {"t": float(r.statistic), "p": float(r.pvalue), "mean_diff": statistics.mean(x - y for x, y in zip(xs, ys))}


def paired_t_manual(xs, ys):
    """path 2 (independent): manual t, p from the t survival function."""
    d = [x - y for x, y in zip(xs, ys)]; n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    if sd == 0:
        return {"t": math.inf if m > 0 else -math.inf, "p": 0.0 if m > 0 else 1.0}
    t = m / (sd / math.sqrt(n))
    return {"t": t, "p": float(stats.t.sf(t, n - 1))}


def ci95(xs):
    n = len(xs); m = statistics.mean(xs); se = statistics.stdev(xs) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return [m - h, m + h]


def seed_acc_from_items(u, alt):
    rows = [r for r in u["results"]["test_heldout"]["per_item"] if r["alt"] == alt]
    return sum(r["pred"] == r["label"] for r in rows) / len(rows)


def main():
    assert not os.path.exists(OUT), "Rule 3"
    units, missing = {}, []
    for dr in DRAWS:
        for a in ARMS:
            for s in SEEDS:
                u = load(dr, a, s)
                if u is None:
                    missing.append("%s_%s_s%d" % (dr, a, s))
                units[(dr, a, s)] = u
    assert not missing, ("missing units", missing)

    acc, twice_ok = {}, True
    for (dr, a, s), u in units.items():
        for alt in ALTS:
            v1 = u["results"]["heldout_" + alt]
            v2 = seed_acc_from_items(u, alt)
            twice_ok &= abs(v1 - v2) < 1e-12
            acc[(dr, a, alt, s)] = v1

    def seeds(dr, a, alt):
        return [acc[(dr, a, alt, s)] for s in SEEDS]

    R = {"per_arm": {}, "gates": {}, "criteria": {}, "diagnostic_Q_COMP_SG": {}, "descriptive": {}}
    for dr in DRAWS:
        for a in ARMS:
            for alt in ALTS:
                xs = seeds(dr, a, alt)
                R["per_arm"]["%s|%s|%s" % (dr, a, alt)] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs),
                                                          "seeds": xs}
    # gates
    conv = [k for k, u in units.items() if u["results"]["train_formI_acc"] < 0.90]
    R["gates"]["G_CONV_unconverged_units"] = ["%s_%s_s%d" % k for k in conv]
    for alt in ALTS:
        g_or = {dr: statistics.mean(seeds(dr, "OP-ORACLE", alt)) for dr in DRAWS}
        g_fl = {dr: ci95(seeds(dr, "Q-CELL", alt)) for dr in DRAWS}
        R["gates"]["G_ORACLE_" + alt] = {"means": g_or, "pass": all(v >= BAR for v in g_or.values())}
        R["gates"]["G_FLOOR_" + alt] = {"ci95": g_fl, "pass": all(lo <= 0.5 <= hi for lo, hi in g_fl.values())}

    def evaluate(arm, alt):
        per = {}
        for dr in DRAWS:
            xs, ys = seeds(dr, arm, alt), seeds(dr, "Q-CELL", alt)
            m = statistics.mean(xs)
            t1, t2 = paired_t(xs, ys), paired_t_manual(xs, ys)
            agree = abs(t1["p"] - t2["p"]) < 1e-9 or (math.isinf(t2["t"]))
            per[dr] = {"mean": m, "bar_ok": m >= BAR, "near_miss": abs(m - BAR) <= NEAR,
                       "vs_qcell": t1, "vs_qcell_twice": t2, "twice_agree": agree,
                       "strict_pass": (m >= BAR) and (t1["p"] < 0.05)}
        p, c = per["primary"], per["confirm"]
        if p["near_miss"]:
            status, val = "PRIMARY NEAR-MISS -> confirm decides", c["strict_pass"]
        else:
            status, val = "both draws", p["strict_pass"] and c["strict_pass"]
        return per, status, bool(val)

    for alt, form in ALTS.items():
        name = ("anti" if alt == "ANTI" else "caus") + "_transfers"
        per, status, val = evaluate("Q-COMP", alt)
        gate_or, gate_fl = R["gates"]["G_ORACLE_" + alt]["pass"], R["gates"]["G_FLOOR_" + alt]["pass"]
        verdict = val if (gate_or and gate_fl) else ("INVALID-CEILING" if not gate_or else "INVALID-FLOOR")
        R["criteria"][name] = {"value": verdict, "raw_bool": val, "rule": status, "per_draw": per}
        twice_ok &= all(d["twice_agree"] for d in per.values())
        per_sg, st_sg, val_sg = evaluate("Q-COMP-SG", alt)
        R["diagnostic_Q_COMP_SG"][name] = {"value_same_bar": val_sg, "rule": st_sg, "per_draw": per_sg,
                                           "note": "DIAGNOSTIC (doc 87 §12.3a); does not enter the verdict"}

    # descriptive
    D = R["descriptive"]
    for dr in DRAWS:
        for a in ["Q-COMP", "Q-COMP-SG", "OP-ORACLE"]:
            fid = {}
            for key in ("routing_fidelity_VII_heldout_roots", "routing_fidelity_VII_seen_roots",
                        "routing_fidelity_IV_heldout_roots", "routing_fidelity_IV_seen_roots"):
                vals = [statistics.mean(units[(dr, a, s)]["extra"][key].values()) for s in SEEDS]
                fid[key] = {"mean": statistics.mean(vals), "seeds": vals}
            for key in ("F_VII_swap_fidelity", "F_IV_swap_fidelity"):
                vals = [units[(dr, a, s)]["extra"][key] for s in SEEDS]
                fid[key] = {"mean": statistics.mean(vals), "seeds": vals}
            D["fidelity|%s|%s" % (dr, a)] = fid
        for a in ARMS:
            seen = {}
            for form in ("I", "VII", "IV"):
                v = []
                for s in SEEDS:
                    rows = [r for r in units[(dr, a, s)]["results"]["test_seen"]["per_item"] if r["form"] == form]
                    v.append(sum(r["pred"] == r["label"] for r in rows) / len(rows))
                seen[form] = statistics.mean(v)
            D["seen_cells|%s|%s" % (dr, a)] = seen
        xs, ys = seeds(dr, "Q-COMP", "ANTI"), seeds(dr, "Q-COMP", "CAUS")
        r = stats.ttest_rel(xs, ys)
        D["anti_minus_caus_QCOMP|%s" % dr] = {"mean_diff": statistics.mean(xs) - statistics.mean(ys),
                                              "t": float(r.statistic), "p_two_sided": float(r.pvalue)}
    lows = {}
    for a in ["Q-COMP", "CL-MLP-FACT"]:
        for n_i in (4, 12, None):
            us = [load("primary", a, s, n_i) for s in SEEDS]
            if any(u is None for u in us):
                lows["%s|nI=%s" % (a, n_i or 40)] = "MISSING"
                continue
            lows["%s|nI=%s" % (a, n_i or 40)] = {alt: statistics.mean(u["results"]["heldout_" + alt] for u in us)
                                                 for alt in ALTS}
    D["lowshot_primary"] = lows
    R["twice_evaluation"] = {"agree": bool(twice_ok),
                             "what": "seed accuracies recomputed from per-item predictions; paired t by scipy and manual"}
    R["doc"] = "87 (59f6621) + §12 (54ddb04)"
    R["provenance"] = C.provenance()
    C.write_new_json(OUT, R)
    for k, v in R["criteria"].items():
        print(k, v["value"], v["rule"], {d: round(x["mean"], 4) for d, x in v["per_draw"].items()},
              {d: round(x["vs_qcell"]["p"], 5) for d, x in v["per_draw"].items()})
    print("gates:", {k: v["pass"] for k, v in R["gates"].items() if isinstance(v, dict)},
          "unconverged:", len(conv), "twice:", twice_ok)


if __name__ == "__main__":
    main()
