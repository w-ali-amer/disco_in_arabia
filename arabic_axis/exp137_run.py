# -*- coding: utf-8 -*-
"""exp137_run.py -- doc 137 (REGISTERED b54fa53): stage E prior sweep.  E1 = K = 2, C = 6, d = 2, ten class priors.
Tuple prior p(a)p(b) | a != b, query uniform over the two positions (exp101 game_rows).  Parts by environment:
  e1q  (qnlp_mac_env, torch): exact classical (exp100 classical_brute), exp100 quantum (gradient), weighted seesaw
  e1u  (qrao_env, cvxpy):     certified level-1 upper bound (exp134 build/solve with weighted payoff)
  e1r  (rac_env):             RAC-tools cross-check on I_0.5 (stage A wrapper)
  e1   (any):                 combine parts, gates, readout -> results_exp137_E1.json (Rule 3)
"""
import itertools, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.join(ROOT, "exp137_logs"); OUT = os.path.join(ROOT, "results_exp137_E1.json")
CLASSES = ["Fem|Dual", "Fem|Plur", "Fem|Sing", "Masc|Dual", "Masc|Plur", "Masc|Sing"]   # sorted, as exp101
X = list(itertools.permutations(range(6), 2))
REF = {"U": (0.8, 0.8828427124746194, 0.8872983346207417), "E-6": (0.9294277836184854, 0.9583845251166967), "E-5": (0.9049223119966019, 0.9466619562500662)}


def priors():
    r101 = json.load(open(os.path.join(ROOT, "results_exp101.json")))["empirical"]
    e6 = np.array([r101["E-6_prodrop"]["classes"][c] for c in CLASSES]); e5 = np.array([r101["E-5_pron_descriptive"]["classes"][c] for c in CLASSES])
    u = np.full(6, 1 / 6); P = {"U": u}
    for t in (0.25, 0.5, 0.75):
        P["I_%g" % t] = (1 - t) * u + t * e6
    P["E-6"] = e6; P["E-5"] = e5
    rank = np.argsort(-e6)                                                       # class indices by E-6 frequency
    for s in (0.5, 1, 2):
        z = np.zeros(6); z[rank] = np.arange(1, 7) ** (-float(s)); P["Zipf_%g" % s] = z / z.sum()
    return {k: v / v.sum() for k, v in P.items()}


def weights(p):
    """w[(xi, pos)] = p(a) p(b) / Z / 2."""
    Z = sum(p[a] * p[b] for a, b in X); return {(xi, pos): p[x[0]] * p[x[1]] / Z / 2 for xi, x in enumerate(X) for pos in (0, 1)}


def seesaw(p, restarts=200, seed=1371):
    """weighted version of exp134 seesaw_k2: complex qubit, projective two-outcome measurements."""
    w = weights(p); rng = np.random.default_rng(seed); best = -1.0
    def value(psi, E):
        return sum(w[(xi, pos)] * float(np.real(psi[xi].conj() @ E[x[pos]][pos] @ psi[xi])) for xi, x in enumerate(X) for pos in (0, 1))
    for _ in range(restarts):
        E = []
        for v in range(6):
            u = rng.standard_normal(2) + 1j * rng.standard_normal(2); u /= np.linalg.norm(u); Pj = np.outer(u, u.conj()); E.append([Pj, np.eye(2) - Pj])
        last = -1.0
        for _ in range(10000):
            psi = []
            for xi, x in enumerate(X):
                O = sum(w[(xi, pos)] * E[x[pos]][pos] for pos in (0, 1)); ev, vec = np.linalg.eigh(O); psi.append(vec[:, -1])
            for v in range(6):
                D = np.zeros((2, 2), complex)
                for xi, x in enumerate(X):
                    if v in x:
                        pos = x.index(v); rho = np.outer(psi[xi], psi[xi].conj()); D += w[(xi, pos)] * rho * (1 if pos == 0 else -1)
                ev, vec = np.linalg.eigh(D); Pj = sum(np.outer(vec[:, i], vec[:, i].conj()) for i in range(2) if ev[i] > 0) if (ev > 0).any() else np.zeros((2, 2), complex)
                E[v] = [Pj, np.eye(2) - Pj]
            val = value(psi, E)
            if abs(val - last) < 1e-13:
                break
            last = val
        best = max(best, val)
    return best


def part_path(name):
    return os.path.join(LOGS, "E1_%s.json" % name)


def write_part(name, obj):
    os.makedirs(LOGS, exist_ok=True); path = part_path(name); assert not os.path.exists(path), "Rule 3"
    with open(path, "x") as f:
        json.dump(obj, f, indent=1)


def e1q():
    import exp100_accounting as A100, exp101_game as G101
    out = {}
    for k, p in priors().items():
        t0 = time.time(); pd = dict(zip(CLASSES, p)); pb = A100.problem(G101.game_rows(CLASSES, pd))
        c = A100.classical_brute(pb, 1); q = A100.quantum(pb, 1); s = seesaw(p)
        out[k] = {"p": list(p), "classical": c, "quantum_grad": q["value"], "G_POVM": q.get("G_POVM"), "seesaw": s, "lower": max(q["value"], s), "wall_s": time.time() - t0}
        print(k, json.dumps(out[k]), flush=True)
    write_part("q", out)


def e1u():
    import exp134_run as E
    out = {}
    for k, p in priors().items():
        w = weights(p); g = {"X": X, "dec": [2] * 6, "pay": [(xi, x[pos], pos, w[(xi, pos)]) for xi, x in enumerate(X) for pos in (0, 1)]}
        r = E.solve(g, 2, 1, ("none",), tag="E1 upper " + k); out[k] = {"upper": r["certified_ub"], "primal": r["primal"], "G_DUAL": r["G_DUAL"], "G_LAGR": r["G_LAGR"]}
    write_part("u", out)


def e1r():
    import exp136_ractools_check as A
    p = list(priors()["I_0.5"]); T = A.bias_tensor(p); c = A.classical(T); s = A.seesaw(T, 60)
    write_part("r", {"prior": "I_0.5", "classical": c["value"], "seesaw": s["value"], "seesaw_converged_flag": s["best_seed"][1], "wall_s": c["wall_s"] + s["wall_s"]})


def e1():
    assert not os.path.exists(OUT), "Rule 3"
    q, u, r = (json.load(open(part_path(n))) for n in ("q", "u", "r")); rows = {}; g = {}
    for k in q:
        lo_c = q[k]["classical"]; rows[k] = {"classical": lo_c, "lower": q[k]["lower"], "upper": u[k]["upper"],
            "adv_interval": [q[k]["lower"] - lo_c, u[k]["upper"] - lo_c]}
        a = rows[k]["adv_interval"]; rows[k]["verdict"] = "survives" if a[0] >= 0.01 else ("dies" if a[1] <= 0 else "open")
    U = rows["U"]
    g["G_REPRO"] = bool(abs(U["classical"] - 0.8) <= 1e-9 and U["lower"] >= REF["U"][1] - 1e-6 and abs(U["upper"] - REF["U"][2]) <= 1e-6)
    g["G_EXP101"] = bool(all(abs(rows[k]["classical"] - REF[k][0]) <= 1e-6 and abs(rows[k]["lower"] - REF[k][1]) <= 1e-4 for k in ("E-6", "E-5")))
    g["G_SANDWICH"] = bool(all(v["classical"] <= v["lower"] + 1e-6 and v["lower"] <= v["upper"] + 1e-6 for v in rows.values()))
    g["G_RACTOOLS_E"] = bool(abs(r["classical"] - rows["I_0.5"]["classical"]) <= 1e-6 and abs(r["seesaw"] - rows["I_0.5"]["lower"]) <= 1e-4)
    g["G_DUAL"] = bool(all(v["G_DUAL"] for v in u.values()))
    m = {k: rows[k]["adv_interval"][0] for k in rows}
    path = ["U", "I_0.25", "I_0.5", "I_0.75", "E-6"]; zp = ["U", "Zipf_0.5", "Zipf_1", "Zipf_2"]
    pred = {"P1_monotone_U_to_E6": all(m[a] >= m[b] - 1e-6 for a, b in zip(path, path[1:])),
            "P1_monotone_Zipf": all(m[a] >= m[b] - 1e-6 for a, b in zip(zp, zp[1:])),
            "P2_survives_all": all(v["verdict"] == "survives" for v in rows.values())}
    res = {"doc": "137 E1", "registered": "b54fa53", "rows": rows, "gates": g, "predictions": pred, "ractools": r,
           "runner_sha256": __import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest()}
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"gates": g, "predictions": pred}, indent=1))
    for k, v in rows.items():
        print("%-9s classical %.6f  lower %.6f  upper %.6f  adv [%.4f, %.4f]  %s" % (k, v["classical"], v["lower"], v["upper"], *v["adv_interval"], v["verdict"]))


if __name__ == "__main__":
    {"e1q": e1q, "e1u": e1u, "e1r": e1r, "e1": e1}[sys.argv[1]]()
