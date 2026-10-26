# -*- coding: utf-8 -*-
"""exp115_run.py -- doc 115 (REGISTERED 1243999): one-pulse (STRICT P1) parity test at C = 7..10.

Reuses exp114_run's reduced objective unchanged.  Per C: classical 1 bit (exp101.analytic, cross-checked by vectorised
exhaustive decoder search); random multistarts (4096 odd C, 8192 even C) + a structured net of 1024 starts (best (C-1)
configuration + one random box); rank-1 signature (2nd singular value of unit g_v < 1e-6) on every start within 1e-6
of the best.  Outcome branches per doc 115 s2.  Writes results_exp115.json (Rule 3).
"""
import itertools, json, math, time
from multiprocessing import Pool
import numpy as np
import exp94_common as C94
import exp101_game as G101
import exp114_run as E

CS = (7, 8, 9, 10)
NET = 1024
MAPS = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])          # decoder: bit -> position (0 = first)


def classical_brute(C):
    D = np.array(list(itertools.product(range(4), repeat=C)), dtype=np.int8)   # (4^C, C)
    tot = np.zeros(len(D))
    for a, b in itertools.permutations(range(C), 2):
        s = [((MAPS[D[:, a], bit] == 0).astype(float) + (MAPS[D[:, b], bit] == 1)) / 2 for bit in (0, 1)]
        tot += np.maximum(*s)
    return float(tot.max() / (C * (C - 1)))


def _ms(a):
    C, seed = a
    v, x = E._ms_job((C, seed))
    return v, x


def _net(a):
    C, seed, xprev = a
    rng = np.random.default_rng(115000 + 1000 * C + seed)
    x0 = np.concatenate([np.asarray(xprev), rng.uniform(0, 2 * math.pi, 2)])
    v, x = E.refine(x0, C)
    return v, x.tolist()


def rank1_sv(x, C):
    g, _ = E.gvecs(np.asarray(x), C)
    gn = np.linalg.norm(g, axis=1); gu = g[gn > 1e-9] / gn[gn > 1e-9, None]
    sv = np.linalg.svd(gu, compute_uv=False)
    return float(sv[1]) if len(sv) > 1 else 0.0


def branch(best, cl):
    if best > cl + 1e-8:
        return "beats"
    if best > cl + 1e-9:
        return "ambiguous"
    return "no point above"


def main():
    t0 = time.time()
    r114 = json.load(open("results_exp114.json"))
    prev_x = r114["multistart"]["6"]["best_x"]
    out = {"doc": "115 (1243999)", "per_C": {}}
    with Pool(10) as pool:
        for C in CS:
            cl_a = G101.analytic(C)["classical"]; cl_b = classical_brute(C)
            n_rand = 4096 if C % 2 else 8192
            rand = pool.map(_ms, [(C, s) for s in range(n_rand)], chunksize=32)
            net = pool.map(_net, [(C, s, prev_x) for s in range(NET)], chunksize=16)
            allr = rand + net
            vals = np.array([v for v, _ in allr]); b = int(vals.argmax()); best = float(vals[b])
            near = [x for v, x in allr if v > best - 1e-6]
            svs = np.array([rank1_sv(x, C) for x in near])
            rec = {"classical_analytic": cl_a, "classical_brute": cl_b, "classical_agree": abs(cl_a - cl_b) < 1e-12,
                   "n_random": n_rand, "n_net": NET, "best": best, "best_source": "random" if b < n_rand else "net",
                   "best_random": float(vals[:n_rand].max()), "best_net": float(vals[n_rand:].max()),
                   "excess": best - cl_b, "branch": branch(best, cl_b), "best_x": allr[b][1],
                   "n_near_best": len(near), "rank1_fraction": float((svs < 1e-6).mean()),
                   "sv2_median": float(np.median(svs)), "geometry": E.geometry(np.asarray(allr[b][1]), C),
                   "quantiles": np.quantile(vals, [0.5, 0.9, 0.99]).tolist()}
            out["per_C"][C] = rec
            prev_x = allr[b][1]
            print("C", C, {k: v for k, v in rec.items() if k not in ("best_x", "geometry")}, int(time.time() - t0), flush=True)
    p = out["per_C"]
    b7, b8 = p[7]["branch"], p[8]["branch"]
    if b7 == "ambiguous":
        verdict = "AMBIGUOUS"
    elif b7 == "beats" and b8 == "no point above" and p[8]["rank1_fraction"] >= 0.95:
        verdict = "TRUE"
    elif b8 == "beats" or b7 == "no point above":
        verdict = "PATTERN BREAKS"
    else:
        verdict = "NOT TRUE (C = 8 %s, rank-1 fraction %.3f)" % (b8, p[8]["rank1_fraction"])
    out["parity_holds"] = verdict
    out["secondary"] = {C: p[C]["branch"] for C in (9, 10)}
    out["wall_s"] = time.time() - t0
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp115.json", out)
    print("VERDICT", verdict, out["secondary"], flush=True)


if __name__ == "__main__":
    main()
