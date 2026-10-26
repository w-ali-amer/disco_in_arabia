# -*- coding: utf-8 -*-
"""exp112_run.py -- doc 112 (REGISTERED c9bd1c5; A1 96ab7f6; judge review + frozen templates §7): deflection Winograd.

For each condition (AR-DEFL, AR-AGREE, AR-SAME, EN-SAME, EN-DEFL) x frame (4) x paraphrase (5): the 4 contexts (slot-1
word i, slot-2 word j) per the frozen mapping; per context the joint over (R1, R2) in {A, B}^2 = softmax of the summed
log-prob of the answer continuation (Qwen3-4B-Instruct-2507 4-bit MLX, the exp109 scorer).  Per frame: CNT, s_odd-2,
Delta on the paraphrase-averaged tables; bootstrap over paraphrases (B = 2000), A3 bias-corrected decision (descriptive).
Gate deflection_dose_response: median CNT AR-AGREE <= AR-DEFL <= AR-SAME AND (AR-DEFL - AR-AGREE) median difference > 0
with one-sided bootstrap p < 0.05.  Run with /opt/homebrew/bin/python3.  Writes results_exp112.json (Rule 3).
"""
import json
import numpy as np
from mlx_lm import load
import exp94_common as C94
import exp109_cbd as X
import exp109_lm as L

COND = ["AR-DEFL", "AR-AGREE", "AR-SAME", "EN-SAME", "EN-DEFL"]
CTX = [(0, 0), (1, 0), (1, 1), (0, 1)]                                  # c1..c4 = (slot1 idx, slot2 idx), frozen


def story(T, fr, cond, i, j):
    lang = "AR" if cond.startswith("AR") else "EN"
    A = fr["A"][lang]; B = fr["B"][cond]
    if lang == "AR":
        s = fr["intro"]["AR"].format(A=A, B=B) + " " + fr["slot1"]["AR"][i] + " " + fr["slot2"]["AR"][j]
    else:
        pr = T["pronouns"][cond]
        def f(x):
            x = x.format(A=A, B=B, P=pr["P"], PP=pr["PP"]); return x[0].upper() + x[1:]
        s = f(fr["intro"]["EN"]) + " " + f(fr["slot1"]["EN"][i]) + " " + f(fr["slot2"]["EN"][j])
    return s, lang, A, B


def joint(model, tok, T, fr, cond, i, j, q):
    s, lang, A, B = story(T, fr, cond, i, j)
    qt = T["question"][lang][q]; cut = qt.index("{X}")
    prefix = s + "\n" + qt[:cut]
    fulls = [s + "\n" + qt.format(X=x, Y=y) for x in (A, B) for y in (A, B)]
    ids, lp = L.logprobs(model, tok, fulls + [prefix]); pid = ids[-1]; v = []
    for k in range(4):
        f = ids[k]
        v.append(L.seq_lp(f, lp[k], len(pid)) if f[:len(pid)] == pid else L.seq_lp(f, lp[k], 1) - L.seq_lp(pid, lp[-1], 1))
    v = np.array(v); p = np.exp(v - v.max()); return (p / p.sum()).reshape(2, 2)


def main():
    T = json.load(open("exp112_templates.json", encoding="utf-8")); model, tok = load(L.MODEL)
    nq = 5; rng = np.random.default_rng(112); res = {}; tabs = {}
    for cond in COND:
        res[cond] = []; tabs[cond] = []
        for fr in T["frames"]:
            per_q = np.array([[joint(model, tok, T, fr, cond, i, j, q) for (i, j) in CTX] for q in range(nq)])   # (q, 4, 2, 2)
            tabs[cond].append(per_q)
            Tm = per_q.mean(0); cnt, sc, dl = X.stats(Tm)
            reps = per_q[rng.integers(0, nq, size=(X.B, nq))].mean(1); cs = X.stats(reps)[0]; bias = cs.mean() - cnt
            res[cond].append({"frame": fr["id"], "cnt": float(cnt), "cnt_bc": float(cnt - bias), "s_odd_minus_2": float(sc), "delta": float(dl),
                              "p": float((1 + ((cs - 2 * bias) <= 0).sum()) / (X.B + 1)), "table": Tm.tolist()})
            print(cond, fr["id"], round(float(cnt), 3), round(float(sc), 3), round(float(dl), 3), flush=True)
    med = {c: float(np.median([r["cnt"] for r in res[c]])) for c in COND}
    diffs = []
    for _ in range(X.B):
        m = {}
        for c in ("AR-DEFL", "AR-AGREE"):
            vals = []
            for per_q in tabs[c]:
                Tm = per_q[rng.integers(0, nq, size=nq)].mean(0); vals.append(float(X.stats(Tm)[0]))
            m[c] = np.median(vals)
        diffs.append(m["AR-DEFL"] - m["AR-AGREE"])
    diffs = np.array(diffs); p_diff = float((1 + (diffs <= 0).sum()) / (X.B + 1))
    gate = bool(med["AR-AGREE"] <= med["AR-DEFL"] <= med["AR-SAME"] and p_diff < 0.05)
    summ = {c: {"median_cnt": med[c], "median_s_odd_minus_2": float(np.median([r["s_odd_minus_2"] for r in res[c]])),
                "median_delta": float(np.median([r["delta"] for r in res[c]])), "n_contextual_bc": int(sum(r["cnt_bc"] > 0 and r["p"] < 0.05 for r in res[c]))}
            for c in COND}
    out = {"doc": "112 (c9bd1c5; A1; §7)", "model": L.MODEL, "summary": summ, "p_diff_defl_minus_agree": p_diff,
           "deflection_dose_response": gate, "per_frame": res, "templates_sha256": C94.sha_file("exp112_templates.json"),
           "provenance": C94.provenance()}
    print(json.dumps({"summary": summ, "p_diff": p_diff, "gate": gate}, ensure_ascii=False), flush=True)
    C94.write_new_json("results_exp112.json", out)


if __name__ == "__main__":
    main()
