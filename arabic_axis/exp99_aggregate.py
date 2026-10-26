# -*- coding: utf-8 -*-
"""exp99_aggregate.py -- SOLE WRITER of results_exp99.json (doc 94 082dc78). Written before the registered run.
Metric: mean single-shot success (exact). beats_bit_2: Q-COMP-2 seed mean >= 0.80 and 95% CI lower > 0.75.
beats_bit_3: Q-COMP-3 95% CI lower > 0.75. free_complex_beats_rebit_3: Q-LOOKUP-3 - R-LOOKUP-3 >= 0.01, paired p < 0.05.
attainment_<arm>: best seed within 0.002 of its registered ceiling. compositional_complex_equals_rebit_3: |best Q-COMP-3 -
best R-COMP-3| <= 0.005. LEAK (halt): any arm > analytic free ceiling + 1e-4. CEILING REVISION (flag): compositional arm >
v3 ceiling + 1e-4. Twice-evaluation: CI and t by scipy and by hand.
"""
import glob, json, math, os, statistics
from scipy import stats
import exp94_common as C

OUT, CK = "results_exp99.json", "exp99_ckpt"
FREE = {2: 0.5 + 1 / (2 * math.sqrt(2)), 3: 0.5 + 1 / (2 * math.sqrt(3))}
V3 = json.load(open("diagnostic_qrac_bounds_v3.json"))["results"]
CEIL = {"Q-COMP-2": V3["compositional_qubit_n2"], "R-COMP-2": V3["compositional_rebit_n2"],
        "Q-COMP-3": V3["compositional_qubit_n3"], "R-COMP-3": V3["compositional_rebit_n3"],
        "Q-LOOKUP-3": FREE[3], "R-LOOKUP-3": 0.5 + (1 + math.sqrt(5)) / 12, "B-1BIT-2": 0.75, "B-1BIT-3": 0.75}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    U = {}
    for p in glob.glob(os.path.join(CK, "*.json")):
        u = json.load(open(p, encoding="utf-8")); U.setdefault(u["arm"], {})[u["seed"]] = u
    arms = sorted(U); assert all(len(U[a]) == 32 for a in arms), {a: len(U[a]) for a in arms}
    S = {a: [U[a][s]["single_shot"] for s in range(32)] for a in arms}
    leak = [(a, s, v) for a in arms for s, v in enumerate(S[a]) if a != "FLOAT-3" and v > FREE[U[a][0]["n"]] + 1e-4]
    assert not leak, ("LEAK: proof violated -> build error", leak)
    revision = {a: max(S[a]) for a in arms if a in CEIL and "COMP" in a and max(S[a]) > CEIL[a] + 1e-4}
    def ci(xs):
        m = statistics.mean(xs); se = statistics.stdev(xs) / math.sqrt(len(xs)); h = stats.t.ppf(0.975, len(xs) - 1) * se
        return [float(m - h), float(m + h)]
    summary = {a: {"mean": statistics.mean(S[a]), "ci95": ci(S[a]), "best": max(S[a]), "worst_seed": min(S[a]),
                   "ceiling": CEIL.get(a), "worst_case_mean": statistics.mean(U[a][s]["worst_case"] for s in range(32)),
                   "argmax_desc": statistics.mean(U[a][s]["argmax_acc_descriptive"] for s in range(32))} for a in arms}
    d = [x - y for x, y in zip(S["Q-LOOKUP-3"], S["R-LOOKUP-3"])]
    md, sd = statistics.mean(d), statistics.stdev(d)
    p1 = float(stats.ttest_rel(S["Q-LOOKUP-3"], S["R-LOOKUP-3"], alternative="greater").pvalue) if sd else (0.0 if md > 0 else 1.0)
    p2 = float(stats.t.sf(md / (sd / math.sqrt(32)), 31)) if sd else p1
    crit = {
        "beats_bit_2": bool(summary["Q-COMP-2"]["mean"] >= 0.80 and summary["Q-COMP-2"]["ci95"][0] > 0.75),
        "beats_bit_3": bool(summary["Q-COMP-3"]["ci95"][0] > 0.75),
        "free_complex_beats_rebit_3": {"value": bool(md >= 0.01 and p1 < 0.05), "mean_diff": float(md), "p": float(p1)},
        "compositional_complex_equals_rebit_3": bool(abs(max(S["Q-COMP-3"]) - max(S["R-COMP-3"])) <= 0.005),
        "attainment": {a: bool(max(S[a]) >= CEIL[a] - 0.002) for a in arms if a in CEIL}}
    Res = {"criteria": crit, "summary": summary, "ceiling_revision_flags": revision,
           "twice_evaluation": {"agree": bool(abs(p1 - p2) < 1e-9)}, "doc": "94 (082dc78)", "provenance": C.provenance()}
    C.write_new_json(OUT, Res)
    print(json.dumps(crit, indent=1, default=float))
    for a in arms:
        s = summary[a]; print("  %-11s mean %.4f CI [%.4f, %.4f] best %.4f ceil %s worst-case %.3f argmax %.3f" % (
            a, s["mean"], s["ci95"][0], s["ci95"][1], s["best"], ("%.4f" % s["ceiling"]) if s["ceiling"] else "-", s["worst_case_mean"], s["argmax_desc"]))
    print("revision flags:", revision)


if __name__ == "__main__":
    main()
