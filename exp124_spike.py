# -*- coding: utf-8 -*-
"""exp124_spike.py -- doc 124 (REGISTERED bd9ba8e): agreement-constrained coreference as QRAO (spike).

Synthetic discourses (exp95-style, flagged synthetic): R in {3,4} referents with agreement classes (gender, number),
at least two referents sharing a class; mentions = one introducing noun per referent + T pro-drop verbs, each verb
agreeing with its gold referent.  Agreement-incompatible pairs cannot corefer (variable removed / constraint).
Affinity (fixed a priori): intro-intro -3; same-class compatible pair +1 + 1/(1+d); otherwise compatible 1/(1+d) - 0.2.
F1 pairwise correlation clustering: x_ij, transitivity constraints (ILP, linear objective).
F2 one-hot assignment y_ic (intro i fixed to cluster i).  F3: the R = 2 binary sub-instance (MaxCut-type).
Per instance: QUBO size, QRAO qubits (QuantumRandomAccessEncoding, max 3 vars/qubit), compression ratio; exact ILP;
QRAO solve (exact relaxed ground state + semideterministic and magic rounding) when qubits <= 14; GW on F3 only.
Closing rule (doc 124): median compression < 2 AND ILP optimal everywhere AND QRAO <= ILP -> close as a note.
Writes results_exp124.json (Rule 3).
"""
import os, json, time, itertools, random
from collections import Counter
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms.qrao import (QuantumRandomAccessEncoding, QuantumRandomAccessOptimizer,
                                                 SemideterministicRounding, MagicRounding)
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.primitives import Sampler

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = [("m", "sg"), ("f", "sg"), ("m", "pl"), ("f", "pl")]
MAXQ = 10                                                  # doc 124 A1: 14 was computationally infeasible (2^14 dense eigensolve per instance)


def instance(rng, R, m):
    while True:
        cls = [rng.choice(CLASSES) for _ in range(R)]
        if len(set(cls)) < R:
            break                                              # at least two referents share a class
    T = m - R
    gold = list(range(R)) + [rng.randrange(R) for _ in range(T)]
    mcls = [cls[g] for g in gold]
    pos = list(range(m))
    w = {}
    for i, j in itertools.combinations(range(m), 2):
        if mcls[i] != mcls[j]:
            continue                                           # agreement-incompatible: cannot corefer
        d = abs(pos[i] - pos[j])
        if i < R and j < R:
            w[(i, j)] = -3.0
        else:
            w[(i, j)] = 1.0 + 1.0 / (1 + d) if (i < R) == (j < R) else 1.0 / (1 + d) - 0.2
    return {"R": R, "m": m, "gold": gold, "cls": mcls, "w": w}


def f1_program(inst):
    m, w = inst["m"], inst["w"]
    qp = QuadraticProgram("f1")
    for (i, j) in w:
        qp.binary_var("x_%d_%d" % (i, j))
    qp.maximize(linear={"x_%d_%d" % k: v for k, v in w.items()})
    nc = 0
    for i, j, k in itertools.combinations(range(m), 3):
        for a, b, c in ((i, j, k), (j, k, i), (i, k, j)):     # x_ab + x_bc - x_ac <= 1 over the three rotations
            e = lambda p, q: tuple(sorted((p, q)))
            ab, bc, ac = e(a, b), e(b, c), e(a, c)
            if ab in w and bc in w:
                lin = {"x_%d_%d" % ab: 1, "x_%d_%d" % bc: 1}
                if ac in w:
                    lin["x_%d_%d" % ac] = -1
                qp.linear_constraint(lin, "<=", 1, name="t%d" % nc); nc += 1
    return qp


def f2_program(inst):
    m, R, w = inst["m"], inst["R"], inst["w"]
    qp = QuadraticProgram("f2")
    for i in range(R, m):
        for c in range(R):
            if inst["cls"][i] == inst["cls"][c]:               # a mention can only join an agreeing referent's cluster
                qp.binary_var("y_%d_%d" % (i, c))
    vars_ = {v.name for v in qp.variables}
    quad, lin = {}, {}
    for (i, j), v in w.items():
        for c in range(R):
            yi = ("y_%d_%d" % (i, c)) if i >= R else (None if i != c else "FIX")
            yj = ("y_%d_%d" % (j, c)) if j >= R else (None if j != c else "FIX")
            if yi is None or yj is None or (yi != "FIX" and yi not in vars_) or (yj != "FIX" and yj not in vars_):
                continue
            if yi == "FIX" and yj == "FIX":
                continue
            if yi == "FIX":
                lin[yj] = lin.get(yj, 0) + v
            elif yj == "FIX":
                lin[yi] = lin.get(yi, 0) + v
            else:
                quad[(yi, yj)] = quad.get((yi, yj), 0) + v
    qp.maximize(linear=lin, quadratic=quad)
    for i in range(R, m):
        names = [n for n in ("y_%d_%d" % (i, c) for c in range(R)) if n in vars_]
        if names:
            qp.linear_constraint({n: 1 for n in names}, "==", 1, name="oh%d" % i)
    return qp


def ilp_opt(qp):
    """Exact optimum of a linear-objective binary program with linear constraints (F1)."""
    n = qp.get_num_vars(); c = -qp.objective.linear.to_array()
    A, lo, hi = [], [], []
    for con in qp.linear_constraints:
        A.append(con.linear.to_array()); sense = con.sense.name
        lo.append(-np.inf if sense == "LE" else con.rhs); hi.append(con.rhs if sense in ("LE", "EQ") else np.inf)
    cons = [LinearConstraint(np.array(A), lo, hi)] if A else []
    r = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
    return float(-r.fun), r.x


def evaluate(qp, x):
    feas = qp.is_feasible(x)
    return float(qp.objective.evaluate(x)), bool(feas)


def gw_maxcut(weights, n, rng, trials=50):
    import cvxpy as cp
    X = cp.Variable((n, n), PSD=True)
    obj = sum(w * (1 - X[i, j]) / 2 for (i, j), w in weights.items())
    cp.Problem(cp.Maximize(obj), [cp.diag(X) == 1]).solve()
    lam, U = np.linalg.eigh((X.value + X.value.T) / 2); V = U * np.sqrt(np.clip(lam, 0, None))   # robust factor (SDP output can be slightly non-PSD)
    best = -np.inf
    for _ in range(trials):
        r = rng.normal(size=n); s = np.sign(V @ r)
        best = max(best, sum(w * (1 - s[i] * s[j]) / 2 for (i, j), w in weights.items()))
    return float(best), float(obj.value)


def qrao_solve(qubo, enc):
    out = {}
    for name, rounding in (("semideterministic", SemideterministicRounding()), ("magic", MagicRounding(sampler=Sampler(options={"shots": 1000, "seed": 0}), seed=0))):
        opt = QuantumRandomAccessOptimizer(min_eigen_solver=NumPyMinimumEigensolver(), rounding_scheme=rounding)
        res = opt.solve(qubo)
        out[name] = list(map(float, res.x))
    return out


def main():
    t0 = time.time()
    out_p = os.path.join(ROOT, "results_exp124.json"); assert not os.path.exists(out_p), "Rule 3"
    rng = random.Random(124); nrng = np.random.default_rng(124)
    conv = QuadraticProgramToQubo()
    recs = []
    for R in (3, 4):
        for m in (6, 8, 10, 12, 14, 16):
            for s in range(50):
                inst = instance(rng, R, m)
                rec = {"R": R, "m": m, "seed": s}
                for fname, builder in (("F1", f1_program), ("F2", f2_program)):
                    qp = builder(inst)
                    if qp.get_num_vars() == 0:
                        rec[fname] = {"n_vars": 0}; continue
                    qubo = conv.convert(qp)
                    enc = QuantumRandomAccessEncoding(max_vars_per_qubit=3); enc.encode(qubo)
                    r = {"n_vars": qp.get_num_vars(), "n_qubo_vars": qubo.get_num_vars(), "qubits": enc.num_qubits,
                         "compression": qubo.get_num_vars() / enc.num_qubits,
                         "decision_vars_per_qubit": qp.get_num_vars() / enc.num_qubits}         # slack variables inflate the first ratio
                    if fname == "F1":
                        opt, xs = ilp_opt(qp); r["ilp_opt"] = opt
                    if enc.num_qubits <= MAXQ:
                        try:
                            sols = qrao_solve(qubo, enc)
                            for rn, x in sols.items():
                                xo = conv.interpret(x)
                                val, feas = evaluate(qp, xo)
                                r["qrao_" + rn] = {"value": val, "feasible": feas}
                        except Exception as e:
                            r["qrao_error"] = str(e)[:200]
                    rec[fname] = r
                if s < 10 and R == 3:                          # F3: the R = 2 binary MaxCut sub-instance
                    sub = [i for i, g in enumerate(inst["gold"]) if g in (0, 1)]
                    W = {}
                    for a, b in itertools.combinations(range(len(sub)), 2):
                        i, j = sub[a], sub[b]
                        k = (min(i, j), max(i, j))
                        W[(a, b)] = -inst["w"][k] if k in inst["w"] else 3.0      # cut weight = -affinity; incompatible -> must cut
                    qp3 = QuadraticProgram("f3")
                    for a in range(len(sub)):
                        qp3.binary_var("z%d" % a)
                    lin, quad = Counter(), Counter()
                    for (a, b), v in W.items():
                        lin["z%d" % a] += v; lin["z%d" % b] += v; quad[("z%d" % a, "z%d" % b)] += -2 * v
                    qp3.maximize(linear=dict(lin), quadratic=dict(quad))
                    enc3 = QuantumRandomAccessEncoding(max_vars_per_qubit=3); enc3.encode(qp3)
                    brute = max(sum(v for (a, b), v in W.items() if bits[a] != bits[b]) for bits in itertools.product((0, 1), repeat=len(sub)))
                    gw, sdp = gw_maxcut(W, len(sub), nrng)
                    r3 = {"n_vars": len(sub), "qubits": enc3.num_qubits, "compression": len(sub) / enc3.num_qubits, "opt": brute, "gw": gw, "sdp_bound": sdp}
                    if enc3.num_qubits <= MAXQ:
                        sols = qrao_solve(qp3, enc3)
                        for rn, x in sols.items():
                            r3["qrao_" + rn] = float(qp3.objective.evaluate(x))
                    rec["F3"] = r3
                recs.append(rec)
                if s % 10 == 9:
                    print("  R", R, "m", m, "inst", s + 1, int(time.time() - t0), flush=True)
            print("R", R, "m", m, "done", int(time.time() - t0), flush=True)
    summ = {}
    for f in ("F1", "F2", "F3"):
        rs = [r[f] for r in recs if f in r and r[f].get("n_vars", 0) > 0]
        comp = [x["compression"] for x in rs]
        summ[f] = {"n": len(rs), "compression_median": float(np.median(comp)), "compression_q90": float(np.quantile(comp, 0.9)),
                   "decision_vars_per_qubit_median": float(np.median([x.get("decision_vars_per_qubit", x["compression"]) for x in rs])),
                   "qubits_median": float(np.median([x["qubits"] for x in rs])), "n_qrao_solved": sum(1 for x in rs if "qrao_semideterministic" in x)}
    f1 = [r["F1"] for r in recs if r["F1"].get("n_vars", 0) > 0]
    ratios = {}
    for rn in ("semideterministic", "magic"):
        v = [(x["qrao_" + rn]["value"] / x["ilp_opt"] if x["ilp_opt"] > 0 else np.nan, x["qrao_" + rn]["feasible"]) for x in f1 if "qrao_" + rn in x]
        ratios[rn] = {"n": len(v), "feasible_frac": float(np.mean([f for _, f in v])) if v else None,
                      "median_ratio_feasible": float(np.nanmedian([a for a, f in v if f])) if any(f for _, f in v) else None,
                      "n_beating_ilp": sum(1 for x in f1 if "qrao_" + rn in x and x["qrao_" + rn]["feasible"] and x["qrao_" + rn]["value"] > x["ilp_opt"] + 1e-9)}
    f3 = [r["F3"] for r in recs if "F3" in r]
    f3s = {"gw_over_opt_median": float(np.median([x["gw"] / x["opt"] for x in f3 if x["opt"] > 0])),
           "qrao_over_opt_median": {rn: float(np.median([x["qrao_" + rn] / x["opt"] for x in f3 if "qrao_" + rn in x and x["opt"] > 0])) for rn in ("semideterministic", "magic")}}
    closing = bool(summ["F1"]["compression_median"] < 2 and summ["F2"]["compression_median"] < 2
                   and all(v["n_beating_ilp"] == 0 for v in ratios.values()))
    out = {"doc": "124 (bd9ba8e)", "synthetic": True, "summary": summ, "F1_qrao_vs_ilp": ratios, "F3": f3s,
           "closing_rule_met": closing, "records": recs, "wall_s": time.time() - t0}
    json.dump(out, open(out_p, "w"), indent=1, default=float)
    print("SUMMARY", json.dumps({k: v for k, v in out.items() if k != "records"}, default=float), flush=True)


if __name__ == "__main__":
    main()
