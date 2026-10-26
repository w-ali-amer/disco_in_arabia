# -*- coding: utf-8 -*-
"""exp109_lm.py -- doc 109 §5 (+A3): LM arm on the SAME primary systems and instances.  Qwen3-4B-Instruct-2507 (MLX
4-bit; timing 0.187 s per batched pass <= 0.3 s registered threshold).  For each stored instance (<= 50 per context):
prompt = "النص: " + <= 15 preceding undiacritised words; 4 candidates "X_mel Y_mel" (top-2 x top-2); score = total
log-prob of the candidate tokens (prefix path if the prompt ids are a prefix of the full ids, else lp(full) - lp(prompt);
path counts registered).  Per-instance joint = softmax over the 4; per-context LM model = mean over instances.
Bootstrap over instances (B = 2000, seed 109), A3 bias-corrected decision + BH.  Usage recomputed on the SAME instances
(gold readings) for a like-for-like comparison.  G-LM: LM argmax marginal accuracy vs majority-melody baseline.
Run with /opt/homebrew/bin/python3 (mlx_lm).  Writes results_exp109_lm.json (Rule 3).
"""
import json, math, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
import exp94_common as C94
import exp109_cbd as X

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
PREFIX = "النص: "


def logprobs(model, tok, texts):
    ids = [tok.encode(t) for t in texts]; L = max(len(s) for s in ids); pad = tok.eos_token_id or 0
    x = mx.array([s + [pad] * (L - len(s)) for s in ids])
    lg = model(x).astype(mx.float32); lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True); mx.eval(lp)
    lp = np.array(lp)
    return ids, lp


def seq_lp(ids, lp, start):
    """sum_{t >= start} log p(ids[t] | ids[:t])."""
    return float(sum(lp[t - 1, ids[t]] for t in range(max(1, start), len(ids))))


def score_instance(model, tok, left, cands, paths):
    prompt = PREFIX + left
    fulls = [prompt + " " + c for c in cands]
    ids, lp = logprobs(model, tok, fulls + [prompt])
    pid = ids[-1]; out = []
    for i in range(len(cands)):
        f = ids[i]
        if f[:len(pid)] == pid:
            out.append(seq_lp(f, lp[i], len(pid))); paths["prefix"] += 1
        else:
            out.append(seq_lp(f, lp[i], 1) - seq_lp(pid, lp[-1], 1)); paths["difference"] += 1
    v = np.array(out); p = np.exp(v - v.max()); return p / p.sum()


def main():
    counts = json.load(open("counts_exp109.json")); H = counts["H"]
    systems = counts["systems_n100"]
    ctxs = sorted({(a, b) for x1, y1, x2, y2 in systems for a, b in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))})
    model, tok = load(MODEL); paths = {"prefix": 0, "difference": 0}
    lmJ, gold, acc = {}, {}, {"lm": 0, "maj": 0, "n": 0}
    for i, (x, y) in enumerate(ctxs):
        inst = counts["instances"]["%s|%s" % (x, y)]
        mx_, my_ = H[x]["top2"], H[y]["top2"]
        cands = [mx_[a] + " " + my_[b] for a in (0, 1) for b in (0, 1)]
        J = []; G = []
        for it in inst:
            p = score_instance(model, tok, it["left"], cands, paths).reshape(2, 2)
            J.append(p); G.append((it["rx"], it["ry"]))
            for marg, g, ax in ((p.sum(1), it["rx"], "x"), (p.sum(0), it["ry"], "y")):
                acc["lm"] += int(np.argmax(marg) == g); acc["maj"] += int(g == 0); acc["n"] += 1
        lmJ[(x, y)] = np.array(J); gold[(x, y)] = G
        if i % 10 == 0:
            print("context", i, "/", len(ctxs), flush=True)
    g_lm = {"lm_acc": acc["lm"] / acc["n"], "majority_acc": acc["maj"] / acc["n"], "n": acc["n"], "paths": paths}
    g_lm["pass"] = bool(g_lm["lm_acc"] > g_lm["majority_acc"])
    print("G-LM", g_lm, flush=True)
    rng = np.random.default_rng(109)

    def lm_system(s):
        x1, y1, x2, y2 = s; cs = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        T = np.array([lmJ[c].mean(0) for c in cs])
        reps = np.empty((X.B, 4, 2, 2))
        for k, c in enumerate(cs):
            J = lmJ[c]; idx = rng.integers(0, len(J), size=(X.B, len(J))); reps[:, k] = J[idx].mean(1)
        return T, reps

    def gold_table(c):
        t = np.zeros((2, 2))
        for a, b in gold[c]:
            t[a, b] += 1
        return t

    lm_res = []
    for s in systems:
        T, reps = lm_system(s)
        cnt, sc, dl = X.stats(T); cs = X.stats(reps)[0]; bias = cs.mean() - cnt
        lm_res.append({"system": s, "cnt": float(cnt), "cnt_bc": float(cnt - bias), "s_odd_minus_2": float(sc), "delta": float(dl),
                       "p": float((1 + ((cs - 2 * bias) <= 0).sum()) / (X.B + 1))})
    ps = np.array([r["p"] for r in lm_res]); m = len(ps); order = np.argsort(ps); thr = X.Q * np.arange(1, m + 1) / m
    passed = ps[order] <= thr; kmax = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool); sig[order[:kmax]] = True
    for r, s_ in zip(lm_res, sig):
        r["bh"] = bool(s_); r["contextual"] = bool(s_ and r["cnt_bc"] > 0)
    sub_tables = [np.array([gold_table(c) for c in ((s[0], s[1]), (s[2], s[1]), (s[2], s[3]), (s[0], s[3]))]) for s in systems]
    usage_sub = X.analyse(sub_tables, rng)
    summ = lambda R: {"n": len(R), "n_contextual": int(sum(r["contextual"] for r in R)), "median_cnt": float(np.median([r["cnt"] for r in R])),
                      "median_s_odd_minus_2": float(np.median([r["s_odd_minus_2"] for r in R])), "median_delta": float(np.median([r["delta"] for r in R])),
                      "max_cnt_bc": float(max(r["cnt_bc"] for r in R))}
    out = {"doc": "109 §5 (+A3)", "model": MODEL, "n_contexts": len(ctxs), "G_LM": g_lm,
           "lm": summ(lm_res), "usage_same_instances": summ(usage_sub),
           "per_system": [{"system": s, "lm_cnt": a["cnt"], "lm_cnt_bc": a["cnt_bc"], "lm_delta": a["delta"], "lm_sodd2": a["s_odd_minus_2"],
                           "usage_cnt": b["cnt"], "usage_delta": b["delta"], "usage_sodd2": b["s_odd_minus_2"]} for s, a, b in zip(systems, lm_res, usage_sub)],
           "corr_cnt_lm_usage": float(np.corrcoef([r["cnt"] for r in lm_res], [r["cnt"] for r in usage_sub])[0, 1]),
           "provenance": C94.provenance()}
    print(json.dumps({k: out[k] for k in ("G_LM", "lm", "usage_same_instances", "corr_cnt_lm_usage")}, ensure_ascii=False), flush=True)
    C94.write_new_json("results_exp109_lm.json", out)


if __name__ == "__main__":
    main()
