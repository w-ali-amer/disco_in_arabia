# -*- coding: utf-8 -*-
"""exp136_ractools_check.py -- doc 136 stage A (REGISTERED 4a25d61): cross-check the K = 2, C = 6 which-referent
game in RAC-tools (Alves-Gigena-Kaniewski 2302.08494, github.com/jedreky/QRAC-tools), run unmodified from a pinned
clone in a separate environment (rac_env).  Encoding: n = 6 characters (one per class), alphabet m = 3 ({absent, first, second}), d = 2;
bias tensor zero off valid ordered pairs and off queries of absent classes.  Priors: uniform; E-6 (exp101).
Deviations forced by licence / speed, applied from outside the package (its files are untouched):
  * MOSEK -> Clarabel: cvxpy.Problem.solve is wrapped so solver="MOSEK" calls Clarabel at tight tolerances.
  * perform_search(method=2) is sharded over processes by wrapping utils.product for the one call with
    repeat = d*n (the decoder enumeration); shard results are merged by max.
Gate G-RACTOOLS: classical within 1e-6 and seesaw within 1e-4 of ours for both priors.
Writes results_exp136_A.json (Rule 3).  Run with the rac_env Python.
"""
import hashlib, itertools, json, multiprocessing as mp, os, subprocess, sys, time
RAC = os.path.expanduser("~/rac_src/QRAC-tools"); sys.path.insert(0, RAC)
import numpy as np, cvxpy as cp
import utils as U

ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "results_exp136_A.json")
PIN = "4093ba84548e85a60fa231c23c1eac048b7667f1"
N, M, D = 6, 3, 2; NSHARD = 9; SEEDS = int(os.environ.get("EXP136_SEEDS", "60"))
CLASSES = ["Fem|Dual", "Fem|Plur", "Fem|Sing", "Masc|Dual", "Masc|Plur", "Masc|Sing"]   # sorted, as exp101 game_rows
OURS = {"uniform": {"classical": 0.8, "qubit": 0.8828427124746194},
        "E-6": {"classical": 0.9294277836184854, "qubit": 0.9583845251166967}}       # results_exp101.json


_solve = cp.Problem.solve
def _solve_no_mosek(self, *a, **k):
    if k.get("solver") == "MOSEK":
        k.pop("mosek_params", None); k["solver"] = "CLARABEL"
        k.update(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500)
    return _solve(self, *a, **k)
cp.Problem.solve = _solve_no_mosek


def bias_tensor(p):
    """alpha[(x, y, b)]: x in {0,1,2}^6 (x_c = 0 absent, 1 first, 2 second), y = queried class, b = answer."""
    if p is None:
        p = [1.0] * N
    valid = {}
    for a, b in itertools.permutations(range(N), 2):
        x = [0] * N; x[a] = 1; x[b] = 2; valid[tuple(x)] = p[a] * p[b]
    tot = sum(valid.values()); T = {}
    for key in itertools.product(itertools.product(range(M), repeat=N), range(N), range(M)):
        x, y, b = key; w = valid.get(x, 0.0)
        T[key] = w / tot / 2 if (w and x[y] != 0 and b == x[y]) else 0.0
    return T


def _shard(args):
    i, T = args; orig = U.product
    def prod(*a, **k):
        it = orig(*a, **k)
        return itertools.islice(it, i, None, NSHARD) if k.get("repeat") == D * N else it
    U.product = prod
    r = U.perform_search(N, D, M, bias_tensor=T, method=2, verbose=False); U.product = orig
    return r["classical value"], r["optimal strategies number"], r["optimal strategy"]


def classical(T):
    t0 = time.time()
    with mp.Pool(NSHARD) as pool:
        parts = pool.map(_shard, [(i, T) for i in range(NSHARD)])
    best = max(v for v, _, _ in parts); B = 1e-7                                  # utils const.BOUND
    return {"value": best, "n_optimal_decoders": sum(c for v, c, _ in parts if abs(v - best) < B),
            "strategy": [list(map(int, s)) for v, _, st in parts if abs(v - best) < B for s in st][:2], "wall_s": time.time() - t0}


def seesaw(T, seeds):
    t0 = time.time(); np.random.seed(13620)
    r = U.perform_seesaw(N, D, seeds, m=M, bias_tensor=T, verbose=False, mosek_accuracy=1e-10)
    return {"value": float(r["optimal value"]), "seeds": seeds, "best_seed": [int(r["best seed number"][0]), bool(r["best seed number"][1])],
            "close_to_largest": int(r["close to largest value"]), "avg_iterations": r["average iterations"], "wall_s": time.time() - t0}


def main():
    assert not os.path.exists(OUT), "Rule 3"
    sha = subprocess.check_output(["git", "-C", RAC, "rev-parse", "HEAD"]).decode().strip(); assert sha == PIN, sha
    assert not subprocess.check_output(["git", "-C", RAC, "status", "--porcelain", "--untracked-files=no"]).strip(), "RAC-tools modified"
    e6 = json.load(open(os.path.join(ROOT, "results_exp101.json")))["empirical"]["E-6_prodrop"]["classes"]
    priors = {"uniform": None, "E-6": [e6[c] for c in CLASSES]}
    res = {"doc": "136 stage A", "registered": "4a25d61", "ractools_sha": sha, "solver": "CLARABEL (MOSEK calls redirected)",
           "versions": {"cvxpy": cp.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
           "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "cases": {}, "gates": {}}
    for name, p in priors.items():
        T = bias_tensor(p); assert abs(sum(T.values()) - 1) < 1e-12
        c = classical(T); print(name, "classical", c, flush=True)
        q = seesaw(T, SEEDS); print(name, "seesaw", q, flush=True)
        dc, dq = abs(c["value"] - OURS[name]["classical"]), abs(q["value"] - OURS[name]["qubit"])
        res["cases"][name] = {"classical": c, "seesaw": q, "ours": OURS[name], "diff_classical": dc, "diff_seesaw": dq}
        res["gates"]["G_RACTOOLS_" + name] = bool(dc <= 1e-6 and dq <= 1e-4)
    res["gates"]["G_RACTOOLS"] = all(res["gates"].values())
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1, default=str)
    print("DONE", res["gates"], flush=True)


if __name__ == "__main__":
    main()
