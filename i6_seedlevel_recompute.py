# -*- coding: utf-8 -*-
"""i6_seedlevel_recompute.py -- doc 82 ledger item I6 / §11.3; doc 70 B2 correction.

Recomputes the pre-doc-59 verdicts that used the item-level sign-flip (anti-conservative:
items correlate within a seed) under doc 70 R3's SEED-LEVEL primary test:
one-sided paired t on seed means (df = n_seeds - 1), with the exact one-sided
seed-block sign-flip p reported beside it (with 5 seeds its floor is 1/32).
Reads committed results JSONs only; edits nothing; writes a NEW file."""
import json, math, itertools, hashlib, os, statistics as st
from scipy import stats

OUT = "results_i6_seedlevel_recompute.json"

def per_seed(path, arm=None, cond=None):
    d = json.load(open(path)); out = {}
    if cond is not None:
        for r in d["per_condition"][cond]:
            out[r["seed"]] = r["test_acc"]
        return out
    def walk(o):
        if isinstance(o, dict):
            if o.get("arm") == arm and "seed" in o and "test_acc" in o:
                out.setdefault(o["seed"], o["test_acc"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d); return out

def test(x, y):
    seeds = sorted(set(x) & set(y)); d = [x[s] - y[s] for s in seeds]
    m = st.mean(d); sd = st.stdev(d); n = len(d)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p_t = float(stats.t.sf(t, n - 1))
    flips = [sum(v * g for v, g in zip(d, sg)) / n for sg in itertools.product([1, -1], repeat=n)]
    p_flip = sum(f >= m - 1e-12 for f in flips) / len(flips)
    return {"seeds": seeds, "diffs": d, "mean_diff": m, "t": t, "df": n - 1,
            "p_one_sided_paired_t": p_t, "p_one_sided_exact_seed_flip": p_flip,
            "n_positive": sum(v > 0 for v in d), "significant_0.05_paired_t": p_t < 0.05}

P2, E44, E48, E55 = ("results_exp43b_phase2.json", "results_exp44.json",
                     "results_exp48.json", "results_exp55.json")
A2 = per_seed(P2, "A2")
C = {
 "exp43b tier_q (A2 vs B1)": (A2, per_seed(P2, "B1"), json.load(open(P2))["verdicts"]["tier_q_passed"]),
 "exp43b tier_q_prime (A3 vs B1)": (per_seed(P2, "A3"), per_seed(P2, "B1"), json.load(open(P2))["verdicts"]["tier_q_prime_passed"]),
 "exp44 tier_q_B4 (A2 vs B4)": (A2, per_seed(E44, "B4"), json.load(open(E44))["verdicts"]["tier_q_B4_passed"]),
 "exp44 tier_q_B5 (A2 vs B5)": (A2, per_seed(E44, "B5"), json.load(open(E44))["verdicts"]["tier_q_B5_passed"]),
 "exp48 tier_m (A4 vs B5)": (per_seed(E48, "A4"), per_seed(E48, "B5"), json.load(open(E48))["verdicts"]["tier_m_passed"]),
 "exp55 random_encoding_fails (E1 vs E0)": (per_seed(E55, cond="E1"), per_seed(E55, cond="E0"), json.load(open(E55))["verdicts"]["random_encoding_fails__E1_vs_E0"]),
 "exp55 richness_helps (E2 vs E1)": (per_seed(E55, cond="E2"), per_seed(E55, cond="E1"), json.load(open(E55))["verdicts"]["richness_helps__E2_vs_E1"]),
 "exp55 encoding_is_semantic (E1 vs EPw)": (per_seed(E55, cond="E1"), per_seed(E55, cond="EPw"), json.load(open(E55))["verdicts"]["encoding_is_semantic__E1_vs_EPw"]),
}
res = {}
for k, (x, y, reg) in C.items():
    r = test(x, y); r["registered_boolean"] = reg
    res[k] = r
    print("%-44s reg=%-5s mean=%+.4f t=%6.2f p_t=%.4f p_flip=%.4f pos=%d/%d"
          % (k, reg, r["mean_diff"], r["t"], r["p_one_sided_paired_t"], r["p_one_sided_exact_seed_flip"], r["n_positive"], len(r["seeds"])))
src = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (P2, E44, E48, E55)}
assert not os.path.exists(OUT), "Rule 3"
json.dump({"purpose": "doc 82 I6 / doc 70 B2: seed-level recompute of pre-doc-59 item-level verdicts",
           "note": "registered booleans stand as computed (Rule 3); this file records what the doc-70 primary test gives. Components that also gate a boolean (e.g. content_earned) are NOT recomputed here.",
           "source_sha256": src, "contrasts": res}, open(OUT, "w"), indent=1)
print("wrote", OUT)
