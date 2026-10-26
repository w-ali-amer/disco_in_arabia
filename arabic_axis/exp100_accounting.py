# -*- coding: utf-8 -*-
"""exp100_accounting.py -- doc 97 §4-§7 (+ §10 A1): exact classical vs explicit quantum, m = 1, 2, under the PADT joint.

Local mix: (x, r, a) from profile_exp100.json. Classical: MILP (HiGHS, gap 0; m=2 falls back to dual bound after 2 h).
Quantum: pure state per x + general POVM per reader class (E_a = S^-1/2 B_a B_a^† S^-1/2), Adam, 64 restarts, float64,
re-evaluated exactly with a POVM validity check (G-POVM).  Remote mix: L-SINGLE (top-2^m mass) + numeric check (G-SINGLE).
§6: empirical 2->1 prior (agreement class; own-Number variant descriptive).  Writes results_exp100.json (Rule 3).
"""
import collections, itertools, json, os, time
import numpy as np
import torch
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import exp94_common as C94

RESTARTS, STEPS = (2, 150) if os.environ.get("SMOKE") else (64, 1500)   # SMOKE: no results written
torch.set_default_dtype(torch.float64)


def problem(rows):
    """rows: list of (x, r, a, weight) -> normalised dict structure."""
    tot = sum(w for *_, w in rows)
    X = sorted({x for x, _, _, _ in rows}); R = sorted({r for _, r, _, _ in rows})
    A = {r: sorted({a for x, rr, a, _ in rows if rr == r}) for r in R}
    P = collections.Counter()
    for x, r, a, w in rows:
        P[(x, r, a)] += w / tot
    return {"X": X, "R": R, "A": A, "P": dict(P)}


def ceiling(pb):
    best = collections.defaultdict(float)
    for (x, r, a), p in pb["P"].items():
        best[(x, r)] = max(best[(x, r)], p)
    per = collections.defaultdict(float)
    for (x, r), p in best.items():
        per[r] += p
    return sum(best.values()), dict(per)


def classical_milp(pb, m, time_limit):
    M = 2 ** m; X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    idx = {}
    def var(k):
        if k not in idx:
            idx[k] = len(idx)
        return idx[k]
    for x in X:
        for s in range(M):
            var(("e", x, s))
    for r in R:
        for s in range(M):
            for a in A[r]:
                var(("d", r, s, a))
    for (x, r, a) in P:
        for s in range(M):
            var(("z", x, r, a, s))
    n = len(idx); c = np.zeros(n); integ = np.zeros(n)
    for k, i in idx.items():
        if k[0] in ("e", "d"):
            integ[i] = 1
        else:
            c[i] = -P[(k[1], k[2], k[3])]
    rows, lo, hi = [], [], []
    for x in X:
        rows.append({idx[("e", x, s)]: 1 for s in range(M)}); lo.append(1); hi.append(1)
    for r in R:
        for s in range(M):
            rows.append({idx[("d", r, s, a)]: 1 for a in A[r]}); lo.append(1); hi.append(1)
    for (x, r, a) in P:
        for s in range(M):
            z = idx[("z", x, r, a, s)]
            rows.append({z: 1, idx[("e", x, s)]: -1}); lo.append(-np.inf); hi.append(0)
            rows.append({z: 1, idx[("d", r, s, a)]: -1}); lo.append(-np.inf); hi.append(0)
    Amat = lil_matrix((len(rows), n))
    for i, row in enumerate(rows):
        for j, v in row.items():
            Amat[i, j] = v
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Amat.tocsr(), lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False})
    val = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", res.fun) if res.x is not None else None
    return {"value": val, "dual_bound": dual, "status": int(res.status), "message": res.message,
            "proved_optimal": res.status == 0, "wall_sec": time.time() - t0, "n_vars": n}


def classical_brute(pb, m):
    """exact by decoder enumeration: every decoder tuple, encoder picks the best message per x."""
    M = 2 ** m; X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    best = 0.0
    for dec in itertools.product(*[list(itertools.product(A[r], repeat=M)) for r in R]):
        D = dict(zip(R, dec)); tot = 0.0
        for x in X:
            tot += max(sum(P.get((x, r, D[r][s]), 0.0) for r in R) for s in range(M))
        best = max(best, tot)
    return best


def quantum(pb, m, seed0=0):
    D = 2 ** m; X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    xi = {x: i for i, x in enumerate(X)}
    W = {r: torch.zeros(len(X), len(A[r])) for r in R}
    for (x, r, a), p in P.items():
        W[r][xi[x], A[r].index(a)] = p
    def povm(B):
        Bc = torch.complex(B[0], B[1]); G = Bc @ Bc.conj().transpose(-1, -2)
        S = G.sum(0); ev, U = torch.linalg.eigh(S)
        Sm = U @ torch.diag(ev.clamp(min=1e-12) ** -0.5).to(U.dtype) @ U.conj().T
        return Sm @ G @ Sm
    def success(psi_ri, Bs):
        psi = torch.complex(psi_ri[0], psi_ri[1]); psi = psi / psi.norm(dim=1, keepdim=True)
        tot = 0.0
        for r in R:
            E = povm(Bs[r])
            pr = torch.einsum("xi,aij,xj->xa", psi.conj(), E, psi).real
            tot = tot + (W[r] * pr).sum()
        return tot
    best = (-1, None)
    for rs in range(RESTARTS):
        g = torch.Generator().manual_seed(100000 + 1000 * m + rs + seed0)
        psi = torch.randn(2, len(X), D, generator=g).requires_grad_()
        Bs = {r: torch.randn(2, len(A[r]), D, D, generator=g).requires_grad_() for r in R}
        opt = torch.optim.Adam([psi] + list(Bs.values()), lr=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        for _ in range(STEPS):
            loss = -success(psi, Bs); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            v = float(success(psi, Bs))
        if v > best[0]:
            best = (v, (psi.detach().clone(), {r: b.detach().clone() for r, b in Bs.items()}))
    # exact re-evaluation + G-POVM
    psi_ri, Bs = best[1]
    with torch.no_grad():
        ok = True; worst_neg = 0.0; worst_sum = 0.0
        for r in R:
            E = povm(Bs[r]); ev = torch.linalg.eigvalsh(E)
            worst_neg = min(worst_neg, float(ev.min()))
            worst_sum = max(worst_sum, float((E.sum(0) - torch.eye(D, dtype=E.dtype)).abs().max()))
        ok = worst_neg >= -1e-10 and worst_sum <= 1e-10
        val = float(success(psi_ri, Bs))
    return {"value": val, "G_POVM": ok, "min_eig": worst_neg, "max_sum_dev": worst_sum, "restarts": RESTARTS, "steps": STEPS}


def top_mass(counter, k):
    tot = sum(counter.values()); return sum(sorted(counter.values(), reverse=True)[:k]) / tot


def main():
    cen = json.load(open("census_exp100.json")); prof = json.load(open("profile_exp100.json"))
    assert prof["G_AGR"]["pass"]
    out = {"doc": "97 (a06fb33) + §10 A1 (92174d6)", "ceiling": {}, "classical": {}, "quantum": {}}
    # ---- local mix
    rows = [(tuple(e["x"]), e["r"], tuple(e["a"]), e["n"]) for e in prof["linked_joint"]]
    pb = problem(rows)
    out["local_sizes"] = {"X": len(pb["X"]), "A": {r: len(v) for r, v in pb["A"].items()}}
    out["ceiling"]["local"] = ceiling(pb)
    # G-MILP: VSO + SVO sub-problem, m = 1
    sub = problem([rw for rw in rows if rw[1] in ("R-VSO", "R-SVO")])
    bm, bb = classical_milp(sub, 1, 600), classical_brute(sub, 1)
    out["G_MILP"] = {"milp": bm["value"], "brute": bb, "pass": abs(bm["value"] - bb) < 1e-12}
    print("G-MILP", out["G_MILP"], flush=True)
    for m in (1, 2):
        out["classical"]["local_m%d" % m] = classical_milp(pb, m, 60 if os.environ.get("SMOKE") else 7200)
        print("classical m", m, out["classical"]["local_m%d" % m], flush=True)
        out["quantum"]["local_m%d" % m] = quantum(pb, m)
        print("quantum m", m, out["quantum"]["local_m%d" % m], flush=True)
    gaps = {}
    for m in (1, 2):
        c = out["classical"]["local_m%d" % m]; cval = c["value"] if c["proved_optimal"] else c["dual_bound"]
        gaps[m] = out["quantum"]["local_m%d" % m]["value"] - cval
    out["gate"] = {"qubit_saving_local": any(g >= 0.05 for g in gaps.values()), "gap_m1": gaps[1], "gap_m2": gaps[2]}
    # ---- remote mix: L-SINGLE + G-SINGLE
    rem = collections.Counter()
    for r in ("R-PRODROP", "R-PRON"):
        for a, c in cen["answer_dist"][r].items():
            rem[a] += c
    out["remote"] = {"answer_dist": dict(rem), "L_SINGLE": {"m%d" % m: top_mass(rem, 2 ** m) for m in (1, 2, 3)},
                     "qubit_saving_remote": False}
    rpb = problem([(a, "REMOTE", a, c) for a, c in rem.items()])
    gs = {}
    for m in (1, 2):
        q = quantum(rpb, m)["value"]; gs["m%d" % m] = {"numeric": q, "lemma": out["remote"]["L_SINGLE"]["m%d" % m],
                                                         "pass": abs(q - out["remote"]["L_SINGLE"]["m%d" % m]) < 1e-6}
    out["G_SINGLE"] = gs
    print("G-SINGLE", gs, flush=True)
    # ---- §6 empirical 2->1
    out["s6"] = {}
    for tag in ("prior_2to1_agr", "prior_2to1_own", "balanced_instrument_check"):
        pr = ({k: 1 for k in ("Masc|Sing", "Fem|Sing", "Masc|nonSing", "Fem|nonSing")} if tag.startswith("balanced")
              else prof[tag])
        rws = []
        for k, c in pr.items():
            g, n = k.split("|")
            rws += [((g, n), "Q-G", g, c), ((g, n), "Q-N", n, c)]
        p2 = problem(rws)
        cl = classical_brute(p2, 1); qu = quantum(p2, 1)
        out["s6"][tag] = {"prior": pr, "classical_exact": cl, "quantum": qu["value"], "G_POVM": qu["G_POVM"],
                          "margin": qu["value"] - cl}
        print("s6", tag, out["s6"][tag], flush=True)
    out["s6"]["empirical_arm_added_to_pasqal_request"] = out["s6"]["prior_2to1_agr"]["margin"] >= 0.05
    out["gates"] = {"G_PARSE": cen["gate_G_PARSE"], "G_AGR": prof["G_AGR"]["pass"], "G_MILP": out["G_MILP"]["pass"],
                    "G_POVM": all(v["G_POVM"] for v in out["quantum"].values()),
                    "G_SINGLE": all(v["pass"] for v in gs.values())}
    out["provenance"] = C94.provenance()
    if os.environ.get("SMOKE"):
        print("SMOKE: not writing", out["gate"], out["gates"]); return
    C94.write_new_json("results_exp100.json", out)
    print(json.dumps({"gate": out["gate"], "gates": out["gates"], "ceiling_local": out["ceiling"]["local"][0],
                      "remote_L_SINGLE": out["remote"]["L_SINGLE"]}, indent=1))


if __name__ == "__main__":
    main()
