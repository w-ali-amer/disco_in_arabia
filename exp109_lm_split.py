# -*- coding: utf-8 -*-
"""exp109_lm_split.py -- doc 109 §12 amendment A4 (descriptive): rerun the LM arm saving per-instance joints; split
Delta into Delta_X (left contents) and Delta_Y (right contents) for usage (same instances), LM-causal (§5 scoring),
and LM-R (right-aware prompt: whole unvowelled phrase shown first).  Also CNT / contextual counts for LM-R (A3 rule).
Run with /opt/homebrew/bin/python3.  Writes results_exp109_lm_split.json (Rule 3).
"""
import json
import numpy as np
from mlx_lm import load
import exp94_common as C94
import exp109_cbd as X
import exp109_lm as L


def score_R(model, tok, left, xf, yf, cands, paths):
    head = L.PREFIX + left + " " + xf + " " + yf + "\nبالتشكيل: " + left
    fulls = [head + " " + c for c in cands]
    ids, lp = L.logprobs(model, tok, fulls + [head]); pid = ids[-1]; out = []
    for i in range(len(cands)):
        f = ids[i]
        if f[:len(pid)] == pid:
            out.append(L.seq_lp(f, lp[i], len(pid))); paths["prefix"] += 1
        else:
            out.append(L.seq_lp(f, lp[i], 1) - L.seq_lp(pid, lp[-1], 1)); paths["difference"] += 1
    v = np.array(out); p = np.exp(v - v.max()); return p / p.sum()


def split(T):
    P = T / T.sum(axis=(-1, -2), keepdims=True)
    ex = P[:, 0, :].sum(-1) - P[:, 1, :].sum(-1); ey = P[:, :, 0].sum(-1) - P[:, :, 1].sum(-1)
    dx = abs(ex[0] - ex[3]) + abs(ex[1] - ex[2]); dy = abs(ey[0] - ey[1]) + abs(ey[2] - ey[3])
    return float(dx), float(dy)


def main():
    counts = json.load(open("counts_exp109.json")); H = counts["H"]; systems = counts["systems_n100"]
    ctxs = sorted({(a, b) for x1, y1, x2, y2 in systems for a, b in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))})
    model, tok = load(L.MODEL); pc = {"prefix": 0, "difference": 0}; pr = {"prefix": 0, "difference": 0}
    J, JR, G = {}, {}, {}
    for i, (x, y) in enumerate(ctxs):
        inst = counts["instances"]["%s|%s" % (x, y)]
        cands = [H[x]["top2"][a] + " " + H[y]["top2"][b] for a in (0, 1) for b in (0, 1)]
        J[(x, y)] = np.array([L.score_instance(model, tok, it["left"], cands, pc).reshape(2, 2) for it in inst])
        JR[(x, y)] = np.array([score_R(model, tok, it["left"], x, y, cands, pr).reshape(2, 2) for it in inst])
        g = np.zeros((2, 2))
        for it in inst:
            g[it["rx"], it["ry"]] += 1
        G[(x, y)] = g
        if i % 10 == 0:
            print("context", i, "/", len(ctxs), flush=True)
    rows = {"usage": [], "lm_causal": [], "lm_R": []}
    rng = np.random.default_rng(109); lmR_res = []
    for s in systems:
        x1, y1, x2, y2 = s; cs = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for name, T in (("usage", np.array([G[c] for c in cs])), ("lm_causal", np.array([J[c].mean(0) for c in cs])),
                        ("lm_R", np.array([JR[c].mean(0) for c in cs]))):
            dx, dy = split(T); cnt, sc, dl = X.stats(T)
            rows[name].append({"system": s, "delta_x": dx, "delta_y": dy, "delta": float(dl), "cnt": float(cnt), "s_odd_minus_2": float(sc)})
        T = np.array([JR[c].mean(0) for c in cs]); reps = np.empty((X.B, 4, 2, 2))
        for k, c in enumerate(cs):
            idx = rng.integers(0, len(JR[c]), size=(X.B, len(JR[c]))); reps[:, k] = JR[c][idx].mean(1)
        cnt = X.stats(T)[0]; csb = X.stats(reps)[0]; bias = csb.mean() - cnt
        lmR_res.append({"cnt_bc": float(cnt - bias), "p": float((1 + ((csb - 2 * bias) <= 0).sum()) / (X.B + 1))})
    ps = np.array([r["p"] for r in lmR_res]); m = len(ps); order = np.argsort(ps); thr = X.Q * np.arange(1, m + 1) / m
    passed = ps[order] <= thr; kmax = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool); sig[order[:kmax]] = True
    n_ctx_R = int(sum(1 for r, s_ in zip(lmR_res, sig) if s_ and r["cnt_bc"] > 0))
    acc = {}
    for name, JJ in (("lm_causal", J), ("lm_R", JR)):
        ok = n = maj = 0
        for c in ctxs:
            inst = counts["instances"]["%s|%s" % c]
            for p, it in zip(JJ[c], inst):
                ok += int(np.argmax(p.sum(1)) == it["rx"]) + int(np.argmax(p.sum(0)) == it["ry"]); maj += int(it["rx"] == 0) + int(it["ry"] == 0); n += 2
        acc[name] = {"acc": ok / n, "majority": maj / n}
    med = lambda name, k: float(np.median([r[k] for r in rows[name]]))
    summary = {name: {k: med(name, k) for k in ("delta_x", "delta_y", "delta", "cnt", "s_odd_minus_2")} for name in rows}
    out = {"doc": "109 §12 A4 (descriptive)", "model": L.MODEL, "summary_medians": summary, "reading_accuracy": acc,
           "lm_R_contextual_bh": n_ctx_R, "paths_causal": pc, "paths_R": pr, "per_system": rows,
           "per_context_joints": {"%s|%s" % c: {"lm_causal": J[c].mean(0).tolist(), "lm_R": JR[c].mean(0).tolist(), "usage_counts": G[c].tolist()} for c in ctxs},
           "provenance": C94.provenance()}
    print(json.dumps({"summary_medians": summary, "reading_accuracy": acc, "lm_R_contextual_bh": n_ctx_R}), flush=True)
    C94.write_new_json("results_exp109_lm_split.json", out)


if __name__ == "__main__":
    main()
