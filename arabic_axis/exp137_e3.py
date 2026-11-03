# -*- coding: utf-8 -*-
"""exp137_e3.py -- doc 137 E3a (REGISTERED ff8eaed): K = 3, C = 6, d = 4, nine class priors.
  e3q  (qnlp_mac_env): quantum lower bound, exp100 quantum(pb, 2) with RESTARTS = 256 (exp113 setting), nine priors in parallel
  e3u  (qrao_env):     certified level-1 upper bound, exp134 solve with d = 4, dec = [3]*6, weighted pay
  e3r  (rac_env):      RAC-tools seesaw at d = 4, alphabet 4, prior I_0.5, 60 seeds (Clarabel redirect)
  e3   (any):          combine with E2's exact classical values, gates, predictions -> results_exp137_E3a.json (Rule 3)
"""
import hashlib, itertools, json, math, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import numpy as np
import exp137_run as R137

LOGS = R137.LOGS; OUT = os.path.join(ROOT, "results_exp137_E3a.json"); K = 3
X = list(itertools.permutations(range(6), K))


def part_path(name):
    return os.path.join(LOGS, "E3a_%s.json" % name)


def write_part(name, obj):
    os.makedirs(LOGS, exist_ok=True)
    with open(part_path(name), "x") as f:
        json.dump(obj, f, indent=1)


def rows(p):
    out = []
    for t in X:
        w = math.prod(p[v] for v in t)
        for pos, v in enumerate(t):
            out.append((t, v, pos, w / K))
    return out


def q_job(item):
    import torch; torch.set_num_threads(1)
    import exp100_accounting as A100
    A100.RESTARTS = 256
    k, p = item; t0 = time.time(); q = A100.quantum(A100.problem(rows(list(p))), 2)
    return k, {"lower": q["value"], "G_POVM": q["G_POVM"], "restarts": 256, "wall_s": time.time() - t0}


def e3q():
    from multiprocessing import Pool
    out = {}
    with Pool(9) as pool:
        for k, r in pool.imap_unordered(q_job, list(R137.priors().items())):
            out[k] = r; print(k, json.dumps(r), flush=True)
    write_part("q", out)


def e3u():
    import exp134_run as E
    out = {}
    for k, p in R137.priors().items():
        Z = sum(math.prod(p[v] for v in t) for t in X)
        pay = [(xi, x[pos], pos, math.prod(p[v] for v in x) / Z / K) for xi, x in enumerate(X) for pos in range(K)]
        g = {"X": X, "dec": [K] * 6, "pay": pay}
        r = E.solve(g, 4, 1, ("none",), tag="E3a upper " + k)
        out[k] = {"upper": r["certified_ub"], "primal": r["primal"], "G_DUAL": r["G_DUAL"], "G_LAGR": r["G_LAGR"]}
        print(k, out[k], flush=True)
    write_part("u", out)


def e3r():
    import exp136_ractools_check as A                                            # applies the MOSEK -> Clarabel redirect
    U = A.U; p = list(R137.priors()["I_0.5"]); N, M, D, seeds = 6, 4, 4, int(os.environ.get("E3R_SEEDS", "60"))
    valid = {}
    for t in X:
        x = [0] * N
        for pos, v in enumerate(t):
            x[v] = pos + 1
        valid[tuple(x)] = math.prod(p[v] for v in t)
    tot = sum(valid.values()); T = {}
    for key in itertools.product(itertools.product(range(M), repeat=N), range(N), range(M)):
        x, y, b = key; w = valid.get(x, 0.0)
        T[key] = w / tot / K if (w and x[y] != 0 and b == x[y]) else 0.0
    assert abs(sum(T.values()) - 1) < 1e-12
    t0 = time.time(); np.random.seed(13730)
    r = U.perform_seesaw(N, D, seeds, m=M, bias_tensor=T, verbose=False, mosek_accuracy=1e-10)
    out = {"prior": "I_0.5", "seesaw": float(r["optimal value"]), "seeds": seeds, "best_seed": [int(r["best seed number"][0]), bool(r["best seed number"][1])],
           "wall_s": time.time() - t0}
    print(json.dumps(out), flush=True); write_part("r", out)


def e3():
    assert not os.path.exists(OUT), "Rule 3"
    q, u = (json.load(open(part_path(n))) for n in ("q", "u"))
    r = json.load(open(part_path("r"))) if os.path.exists(part_path("r")) else None
    cl = json.load(open(os.path.join(ROOT, "results_exp137_E2.json")))["k3_classical_exact"]
    rows_, g = {}, {}
    for k in q:
        lo = q[k]["lower"]
        if k == "I_0.5" and r is not None:
            lo = max(lo, r["seesaw"])
        c = cl[k]; a = [lo - c, u[k]["upper"] - c]
        rows_[k] = {"classical": c, "lower": lo, "lower_exp100": q[k]["lower"], "upper": u[k]["upper"], "gap": u[k]["upper"] - lo, "adv_interval": a,
                    "verdict": "survives" if a[0] >= 0.01 else ("dies" if a[1] <= 0 else "open")}
    Uo = rows_["U"]
    g["G_REPRO"] = bool(abs(Uo["classical"] - 13 / 18) <= 1e-12 and Uo["lower_exp100"] >= 0.7781249 - 1e-4 and abs(Uo["upper"] - 0.8201978) <= 1e-6)
    g["G_SANDWICH"] = bool(all(v["classical"] <= v["lower"] + 1e-6 and v["lower"] <= v["upper"] + 1e-6 for v in rows_.values()))
    g["G_DUAL"] = bool(all(v["G_DUAL"] for v in u.values()))
    g["G_POVM"] = bool(all(v["G_POVM"] for v in q.values()))
    if r is not None:
        g["G_RACTOOLS_E3"] = bool(r["seesaw"] <= u["I_0.5"]["upper"] + 1e-6 and q["I_0.5"]["lower"] <= u["I_0.5"]["upper"] + 1e-6)
    m = {k: v["adv_interval"][0] for k, v in rows_.items()}
    path = ["U", "I_0.25", "I_0.5", "I_0.75", "E-6"]; zp = ["U", "Zipf_0.5", "Zipf_1", "Zipf_2"]
    pred = {"P3a_monotone_U_to_E6": all(m[a] >= m[b] - 1e-6 for a, b in zip(path, path[1:])),
            "P3a_monotone_Zipf": all(m[a] >= m[b] - 1e-6 for a, b in zip(zp, zp[1:])),
            "P3b_K3_beats_K2_at_E6": rows_["E-6"]["lower"] >= 0.8066, "P3b_E6_survives": rows_["E-6"]["verdict"] == "survives",
            "P3c_gap_ratio_E6_over_U": rows_["E-6"]["gap"] / Uo["gap"], "P3c_holds": rows_["E-6"]["gap"] / Uo["gap"] >= 0.1}
    extra = {}
    if r is not None:
        d = r["seesaw"] - q["I_0.5"]["lower"]; extra = {"ractools_minus_exp100": d, "flag_gt_1e-3": abs(d) > 1e-3}
    res = {"doc": "137 E3a", "registered": "ff8eaed", "rows": rows_, "gates": g, "predictions": pred, "ractools": r, "ractools_check": extra,
           "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"gates": g, "predictions": pred, "ractools_check": extra}, indent=1))
    for k, v in rows_.items():
        print("%-9s classical %.6f  lower %.6f  upper %.6f  gap %.2e  adv [%.4f, %.4f]  %s" % (k, v["classical"], v["lower"], v["upper"], v["gap"], *v["adv_interval"], v["verdict"]))


if __name__ == "__main__":
    {"e3q": e3q, "e3u": e3u, "e3r": e3r, "e3": e3}[sys.argv[1]]()
