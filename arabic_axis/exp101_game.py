# -*- coding: utf-8 -*-
"""exp101_game.py -- doc 98 (REGISTERED 8a02346): "which one am I?" -- pronoun resolution by agreement as a
one-qubit prepare-and-measure game.  Alice: ordered pair (c1, c2), c1 != c2 (symmetric prior); Bob: v uniform in
{c1, c2}; output first/second.  Classical exact (brute force over decoders), qubit + rebit by the general optimiser
(exp100's pure-state + general-POVM parameterisation), T2 analytic values, T3 two-bit construction.
Writes results_exp101.json (Rule 3).
"""
import itertools, json, math, os
import torch
import exp94_common as C94
import exp100_accounting as A100

torch.set_default_dtype(torch.float64)


def game_rows(classes, p):
    """rows (x=(a,b), r=v, answer, weight) for the symmetric pair prior p(a)p(b) | a != b, v uniform in pair."""
    rows = []
    for a, b in itertools.permutations(classes, 2):
        w = p[a] * p[b]
        rows.append(((a, b), a, "first", w / 2)); rows.append(((a, b), b, "second", w / 2))
    return rows


def rebit(pb, restarts=A100.RESTARTS, steps=A100.STEPS):
    """real states on the Bloch x-z circle + real binary POVMs E = S^-1/2 B B^T S^-1/2 (G-POVM checked)."""
    X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    xi = {x: i for i, x in enumerate(X)}
    W = {r: torch.zeros(len(X), len(A[r])) for r in R}
    for (x, r, a), pr in P.items():
        W[r][xi[x], A[r].index(a)] = pr
    def povm(B):
        G = B @ B.transpose(-1, -2); ev, U = torch.linalg.eigh(G.sum(0))
        Sm = U @ torch.diag(ev.clamp(min=1e-12) ** -0.5) @ U.T
        return Sm @ G @ Sm
    def success(psi, Bs):
        psi = psi / psi.norm(dim=1, keepdim=True); tot = 0.0
        for r in R:
            tot = tot + (W[r] * torch.einsum("xi,aij,xj->xa", psi, povm(Bs[r]), psi)).sum()
        return tot
    best = (-1, None)
    for rs in range(restarts):
        g = torch.Generator().manual_seed(101000 + rs)
        psi = torch.randn(len(X), 2, generator=g).requires_grad_()
        Bs = {r: torch.randn(len(A[r]), 2, 2, generator=g).requires_grad_() for r in R}
        opt = torch.optim.Adam([psi] + list(Bs.values()), lr=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
        for _ in range(steps):
            loss = -success(psi, Bs); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            v = float(success(psi, Bs))
        if v > best[0]:
            best = (v, psi.detach().clone(), {r: b.detach().clone() for r, b in Bs.items()})
    with torch.no_grad():
        neg = min(float(torch.linalg.eigvalsh(povm(b)).min()) for b in best[2].values())
        dev = max(float((povm(b).sum(0) - torch.eye(2)).abs().max()) for b in best[2].values())
        val = float(success(best[1], best[2]))
    return {"value": val, "G_POVM": neg >= -1e-10 and dev <= 1e-10}


def analytic(C):
    pc = 0.5 + (C // 2) * ((C + 1) // 2) / (C * (C - 1))
    q = {2: 1.0, 3: 0.5 + math.sqrt(3) / 4, 4: 0.5 + math.sqrt(8 / 3) / 4, 6: 0.5 + (2 * math.sqrt(2) + 1) / 10}
    poly = sum(2 * math.sin(math.pi * k / C) for i in range(C) for k in range(1, C)) / 2  # ordered -> unordered pairs
    rb = 0.5 + poly / (2 * C * (C - 1))
    return {"classical": pc, "qubit": q.get(C), "rebit": rb, "stolarsky_upper": 0.5 + C / (3 * (C - 1))}


def two_bit_check(C):
    """T3: classes -> distinct 2-subsets of {1..4}; colour k in S_a minus S_b; 'first' iff k in S_v. Returns success."""
    S = list(itertools.combinations(range(4), 2))[:C]
    ok = tot = 0
    for a, b in itertools.permutations(range(C), 2):
        k = sorted(set(S[a]) - set(S[b]))[0]
        for v, ans in ((a, "first"), (b, "second")):
            out = "first" if k in S[v] else "second"
            ok += out == ans; tot += 1
    return ok / tot


def main():
    out = {"doc": "98 (8a02346)", "uniform": {}, "gates": {}}
    gc, gr, g2, gp = True, True, True, True
    for C in range(2, 7):
        cl = list(range(C)); pb = A100.problem(game_rows(cl, {c: 1.0 for c in cl}))
        brute = A100.classical_brute(pb, 1); q = A100.quantum(pb, 1); rb = rebit(pb); an = analytic(C); tb = two_bit_check(C)
        out["uniform"][C] = {"classical_exact": brute, "qubit": q["value"], "rebit": rb["value"], "analytic": an,
                             "two_bit_construction": tb, "margin_qubit": q["value"] - brute}
        gc &= abs(brute - an["classical"]) < 1e-12
        if an["qubit"] is not None:
            gr &= abs(q["value"] - an["qubit"]) < 1e-6
        gr &= abs(rb["value"] - an["rebit"]) < 1e-6
        g2 &= tb == 1.0; gp &= q["G_POVM"] and rb["G_POVM"]
        print(C, out["uniform"][C], flush=True)
    c5 = out["uniform"][5]
    out["C5_note"] = {"attained": c5["qubit"], "stolarsky_upper": c5["analytic"]["stolarsky_upper"]}
    cen = json.load(open("census_exp100.json"))
    out["empirical"] = {}
    for tag, cls in (("E-6_prodrop", "R-PRODROP"), ("E-5_pron_descriptive", "R-PRON")):
        dist = {k: v for k, v in cen["answer_dist"][cls].items() if "∅" not in k}
        tot = sum(dist.values()); p = {k: v / tot for k, v in dist.items()}
        pb = A100.problem(game_rows(sorted(p), p))
        brute = A100.classical_brute(pb, 1); q = A100.quantum(pb, 1)
        gp &= q["G_POVM"]
        out["empirical"][tag] = {"classes": p, "classical_exact": brute, "qubit": q["value"], "margin": q["value"] - brute}
        print(tag, out["empirical"][tag], flush=True)
    out["gates"] = {"G_CLASS": gc, "G_REDUCE": gr, "G_2BIT": g2, "G_POVM": gp}
    out["decisions"] = {"which_referent_uniform": out["uniform"][6]["margin_qubit"] >= 0.05,
                        "which_referent_empirical": out["empirical"]["E-6_prodrop"]["margin"] >= 0.05}
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp101.json", out)
    print(json.dumps({"gates": out["gates"], "decisions": out["decisions"]}, indent=1))


if __name__ == "__main__":
    main()
