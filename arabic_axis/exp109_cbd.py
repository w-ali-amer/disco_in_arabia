# -*- coding: utf-8 -*-
"""exp109_cbd.py -- doc 109 (REGISTERED b9c4aff; A1 030e0f3; A2): CbD contextuality of rank-4 cyclic systems of
Arabic lexical homographs in real usage (Tashkeela bigrams).  Gates in order with STOP: G-CBD, G-SIGN, G-NULL, G-POWER.

System (X1, Y1, X2, Y2); contents q1=X1, q2=Y1, q3=X2, q4=Y2; contexts c1=(X1Y1) c2=(X2Y1) c3=(X2Y2) c4=(X1Y2).
Table t[a][b]: a = left reading (0 -> +1, 1 -> -1), b = right reading.  CNT = s_odd(rho) - Delta - 2 (Kujala-Dzhafarov).
Bootstrap B = 2000 multinomial per context (seed 109); CNT_bc = 2 CNT - mean(CNT*); p = (1 + #{CNT* <= 0})/(B + 1);
BH q = 0.05; declared contextual iff BH-significant AND CNT_bc > 0.  Writes results_exp109_usage.json (Rule 3).
"""
import json, math, sys
import numpy as np
import exp94_common as C94

B, Q, SEED = 2000, 0.05, 109


def stats(T):
    """T: (..., 4 contexts, 2, 2) probabilities or counts -> (CNT, s_odd-2, Delta)."""
    T = np.asarray(T, float); N = T.sum(axis=(-1, -2), keepdims=True); P = T / np.where(N == 0, 1, N)
    ex = P[..., 0, :].sum(-1) - P[..., 1, :].sum(-1)                  # E[left]  per context
    ey = P[..., :, 0].sum(-1) - P[..., :, 1].sum(-1)                  # E[right] per context
    rho = P[..., 0, 0] + P[..., 1, 1] - P[..., 0, 1] - P[..., 1, 0]
    a = np.abs(rho); nneg = (rho < 0).sum(-1)
    s_odd = np.where(nneg % 2 == 1, a.sum(-1), a.sum(-1) - 2 * a.min(-1))
    # contexts: c1=(X1,Y1) c2=(X2,Y1) c3=(X2,Y2) c4=(X1,Y2)
    delta = (np.abs(ex[..., 0] - ex[..., 3]) + np.abs(ey[..., 0] - ey[..., 1]) +
             np.abs(ex[..., 1] - ex[..., 2]) + np.abs(ey[..., 2] - ey[..., 3]))
    return s_odd - delta - 2, s_odd - 2, delta


def boot(T, rng, b=B):
    T = np.asarray(T, float); reps = np.empty((b, 4, 2, 2))
    for k in range(4):
        n = int(T[k].sum()); p = T[k].ravel() / n
        reps[:, k] = rng.multinomial(n, p, size=b).reshape(b, 2, 2)
    return reps


def analyse(tables, rng):
    out = []
    for T in tables:
        cnt, sc, dl = stats(T)
        cs = stats(boot(T, rng))[0]; bias = cs.mean() - cnt                # A3: net bias (Delta deflation vs s_odd inflation)
        out.append({"cnt": float(cnt), "cnt_bc": float(cnt - bias), "s_odd_minus_2": float(sc), "delta": float(dl), "bias": float(bias),
                    "p_raw": float((1 + (cs <= 0).sum()) / (B + 1)),
                    "p": float((1 + ((cs - 2 * bias) <= 0).sum()) / (B + 1))})         # A3: bias-corrected percentile test
    ps = np.array([o["p"] for o in out]); m = len(ps)
    if m:
        order = np.argsort(ps); thr = Q * (np.arange(1, m + 1)) / m
        passed = ps[order] <= thr; kmax = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
        sig = np.zeros(m, bool); sig[order[:kmax]] = True
        for o, s in zip(out, sig):
            o["bh"] = bool(s); o["contextual"] = bool(s and o["cnt_bc"] > 0)
    return out


def shuffle_table(T, rng):
    """independent re-pairing within a context: hypergeometric 2x2 with the observed margins."""
    T = np.asarray(T, int); n = T.sum(); r0 = T[0].sum(); c0 = T[:, 0].sum()
    a = rng.hypergeometric(c0, n - c0, r0)
    return np.array([[a, r0 - a], [c0 - a, n - r0 - c0 + a]])


def gates(real_tables, rng):
    g = {}
    pr = np.array([[[.5, 0], [0, .5]]] * 3 + [[[0, .5], [.5, 0]]])
    det = np.array([[[1, 0], [0, 0]]] * 4)
    prod_ok = True
    for _ in range(1000):
        m = rng.uniform(-1, 1, size=(4,))                              # consistent marginals per content
        mk = [(m[0], m[1]), (m[2], m[1]), (m[2], m[3]), (m[0], m[3])]
        T = np.array([np.outer([(1 + x) / 2, (1 - x) / 2], [(1 + y) / 2, (1 - y) / 2]) for x, y in mk])
        prod_ok &= bool(stats(T)[0] <= 1e-12)
        mi = rng.uniform(-1, 1, size=(4, 2))                           # inconsistent marginals too
        T2 = np.array([np.outer([(1 + x) / 2, (1 - x) / 2], [(1 + y) / 2, (1 - y) / 2]) for x, y in mi])
        prod_ok &= bool(stats(T2)[0] <= 1e-12)
    g["G_CBD"] = {"pr_box": float(stats(pr)[0]), "deterministic": float(stats(det)[0]), "product_noncontextual": prod_ok,
                  "pass": bool(abs(stats(pr)[0] - 2) < 1e-12 and stats(det)[0] <= 0 and prod_ok)}
    ok = True
    for _ in range(1000):
        T = rng.dirichlet(np.ones(4), size=4).reshape(4, 2, 2); c0 = stats(T)[0]
        k = rng.integers(4)                                            # flip one content's labels
        T2 = T.copy()
        if k == 0: T2[0] = T2[0][::-1]; T2[3] = T2[3][::-1]
        if k == 1: T2[0] = T2[0][:, ::-1]; T2[1] = T2[1][:, ::-1]
        if k == 2: T2[1] = T2[1][::-1]; T2[2] = T2[2][::-1]
        if k == 3: T2[2] = T2[2][:, ::-1]; T2[3] = T2[3][:, ::-1]
        ok &= abs(stats(T2)[0] - c0) < 1e-12
    g["G_SIGN"] = {"pass": bool(ok)}
    nulls = [np.array([shuffle_table(t, rng) for t in T]) for T in real_tables]
    res = analyse(nulls, rng); fpr = float(np.mean([r["contextual"] for r in res])) if res else 0.0
    g["G_NULL"] = {"n_systems": len(res), "fpr": fpr, "pass": bool(fpr <= 0.05)}
    # A3 (reviewer 2): signalling boundary, CNT = 0 exactly: c1-c3 X = Y uniform; c4 X1 = +1, Y2 = -1
    Pb = np.array([[[.5, 0], [0, .5]]] * 3 + [[[0, 1.0], [0, 0]]])
    assert abs(stats(Pb)[0]) < 1e-12
    bnd = {}
    for n in (100, 50):
        sims = [np.array([rng.multinomial(n, Pb[k].ravel()).reshape(2, 2) for k in range(4)]) for _ in range(300)]
        r = analyse(sims, rng)
        bnd["n%d" % n] = {"fpr_bh": float(np.mean([x["contextual"] for x in r])), "fpr_pertest": float(np.mean([x["p"] < 0.05 and x["cnt_bc"] > 0 for x in r]))}
    g["G_BOUNDARY_SIGNAL"] = {**bnd, "pass": bool(all(v["fpr_pertest"] <= 0.05 for v in bnd.values()))}
    power = {}
    for n in (100, 50):
        for name, w in (("pr_box", 1.0), ("moderate_cnt0.5", 0.625)):
            planted = []
            for _ in range(200):
                P = w * pr + (1 - w) * np.full((4, 2, 2), .25)
                planted.append(np.array([rng.multinomial(n, P[k].ravel()).reshape(2, 2) for k in range(4)]))
            null = [np.array([rng.multinomial(n, np.full(4, .25)).reshape(2, 2) for k in range(4)]) for _ in range(200)]
            r = analyse(planted + null, rng)
            power["n%d_%s" % (n, name)] = float(np.mean([x["contextual"] for x in r[:200]]))
    detectable = None
    for w in np.arange(0.55, 1.001, 0.025):
        planted = [np.array([rng.multinomial(100, (w * pr + (1 - w) * np.full((4, 2, 2), .25))[k].ravel()).reshape(2, 2) for k in range(4)]) for _ in range(200)]
        null = [np.array([rng.multinomial(100, np.full(4, .25)).reshape(2, 2) for k in range(4)]) for _ in range(200)]
        r = analyse(planted + null, rng)
        if np.mean([x["contextual"] for x in r[:200]]) >= 0.8:
            detectable = float(4 * w - 2); break
    g["G_POWER"] = {**power, "cnt_detectable_at_0.8_n100": detectable,
                    "pass": bool(power["n100_pr_box"] >= 0.95 and power["n50_pr_box"] >= 0.95),
                    "note": None if power["n100_moderate_cnt0.5"] >= 0.8 else "a negative is only informative for CNT >= %s" % detectable}
    return g


def tables_for(counts, s):
    x1, y1, x2, y2 = s
    t = counts["tables"]
    return np.array([t["%s|%s" % (x1, y1)], t["%s|%s" % (x2, y1)], t["%s|%s" % (x2, y2)], t["%s|%s" % (x1, y2)]], float)


def main():
    counts = json.load(open("counts_exp109.json")); rng = np.random.default_rng(SEED)
    out = {"doc": "109 (b9c4aff; A1; A2)", "n_systems_n100": len(counts["systems_n100"]), "n_systems_n50": len(counts["systems_n50"])}
    prim = [tables_for(counts, s) for s in counts["systems_n100"]]
    g = gates(prim if prim else [tables_for(counts, s) for s in counts["systems_n50"]], rng)
    out["gates"] = g; print(json.dumps(g), flush=True)
    n50 = [tables_for(counts, s) for s in counts["systems_n50"]]
    r50 = analyse([np.array([shuffle_table(t, rng) for t in T]) for T in n50], rng)
    g["G_NULL_n50_descriptive"] = {"n_systems": len(r50), "fpr": float(np.mean([r["contextual"] for r in r50])) if r50 else None}
    print("G_NULL_n50", g["G_NULL_n50_descriptive"], flush=True)
    for k in ("G_CBD", "G_SIGN", "G_NULL", "G_BOUNDARY_SIGNAL", "G_POWER"):
        if not g[k]["pass"]:
            out["stopped_at"] = k; out["provenance"] = C94.provenance()
            C94.write_new_json("results_exp109_usage.json", out); print("STOPPED", k); sys.exit(0)
    for tag, key in (("n100_primary", "systems_n100"), ("n50_descriptive", "systems_n50")):
        sy = counts[key]; res = analyse([tables_for(counts, s) for s in sy], rng)
        for r, s in zip(res, sy):
            r["system"] = s
        out[tag] = {"n": len(res), "n_contextual": int(sum(r["contextual"] for r in res)),
                    "frac_contextual": float(np.mean([r["contextual"] for r in res])) if res else None,
                    "median_cnt": float(np.median([r["cnt"] for r in res])) if res else None,
                    "median_s_odd_minus_2": float(np.median([r["s_odd_minus_2"] for r in res])) if res else None,
                    "median_delta": float(np.median([r["delta"] for r in res])) if res else None,
                    "max_cnt_bc": float(max(r["cnt_bc"] for r in res)) if res else None,
                    "systems": res}
        print(tag, {k: v for k, v in out[tag].items() if k != "systems"}, flush=True)
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp109_usage.json", out)


if __name__ == "__main__":
    main()
