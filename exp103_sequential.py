# -*- coding: utf-8 -*-
"""exp103_sequential.py -- doc 100 (REGISTERED 8a02346): two readers share one qubit (sequential QRAC), theory.

Reader 1: Lueders instrument of sharpness eta along its class axis n_q1 (binary answer; outcome + -> A1[q1][0]).
Reader 2 (knows neither q1 nor outcome) sees r' = (n.r) n + sqrt(1-eta^2) (r - (n.r) n) and applies a GENERAL POVM
per class.  For each eta in {0, .05, ..., 1}: maximise min(P1 - C1, P2 - C2) (C = classical single-reader optimum,
bits are copyable so the classical corner is (C1, C2)); 16 restarts, float64.  Arms S1 (2->1), S2 (which-referent
C = 3, 6), S3 (nested VSO gender -> pronoun gender+number).  Writes results_exp103.json (Rule 3).
"""
import itertools, json, math
import torch
import exp94_common as C94

torch.set_default_dtype(torch.float64)
ETAS = [round(0.05 * k, 2) for k in range(21)]
RESTARTS, STEPS = 16, 800
PAULI = torch.stack([torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
                     torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
                     torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)])


def arm_S1():
    X = list(itertools.product((0, 1), repeat=2)); items = []
    for x in X:
        for q1 in (0, 1):
            for q2 in (0, 1):
                items.append((x, q1, q2, x[q1], x[q2], 1 / 16))
    return X, {q: (0, 1) for q in (0, 1)}, {q: (0, 1) for q in (0, 1)}, items, (0.75, 0.75)


def arm_S2(C):
    X = list(itertools.permutations(range(C), 2)); items = []
    w = 1 / (len(X) * 4)
    for (a, b) in X:
        for v1 in (a, b):
            for v2 in (a, b):
                items.append(((a, b), v1, v2, "first" if v1 == a else "second", "first" if v2 == a else "second", w))
    pc = 0.5 + (C // 2) * ((C + 1) // 2) / (C * (C - 1))
    return X, {v: ("first", "second") for v in range(C)}, {v: ("first", "second") for v in range(C)}, items, (pc, pc)


def arm_S3():
    X = list(itertools.product(("M", "F"), ("S", "D"))); items = []
    for x in X:
        items.append((x, "VSO", "PRON", x[0], x, 1 / 4))
    return X, {"VSO": ("M", "F")}, {"PRON": tuple(X)}, items, (1.0, 0.5)


def solve(arm, eta, seed):
    X, A1, A2, items, corner = arm
    Q1, Q2 = sorted(A1, key=str), sorted(A2, key=str)
    g = torch.Generator().manual_seed(103000 + seed)
    rv = torch.randn(len(X), 3, generator=g).requires_grad_()
    nv = torch.randn(len(Q1), 3, generator=g).requires_grad_()
    B = {q: torch.randn(2, len(A2[q]), 2, 2, generator=g).requires_grad_() for q in Q2}
    xi = {x: i for i, x in enumerate(X)}
    s = math.sqrt(max(0.0, 1 - eta ** 2))
    idx = [(xi[x], Q1.index(q1), q2, 1.0 if a1 == A1[q1][0] else -1.0, A2[q2].index(a2), w) for x, q1, q2, a1, a2, w in items]
    def povm(Bq):
        Bc = torch.complex(Bq[0], Bq[1]); G = Bc @ Bc.conj().transpose(-1, -2)
        ev, U = torch.linalg.eigh(G.sum(0)); Sm = U @ torch.diag(ev.clamp(min=1e-12) ** -0.5).to(U.dtype) @ U.conj().T
        return Sm @ G @ Sm
    XI = torch.tensor([i[0] for i in idx]); QI = torch.tensor([i[1] for i in idx])
    SG = torch.tensor([i[3] for i in idx]); W = torch.tensor([i[5] for i in idx])
    groups = {q: [k for k, i in enumerate(idx) if i[2] == q] for q in Q2}
    AI = {q: torch.tensor([idx[k][4] for k in groups[q]]) for q in Q2}
    GI = {q: torch.tensor(groups[q]) for q in Q2}
    def evaluate():
        r = rv / rv.norm(dim=1, keepdim=True); n = nv / nv.norm(dim=1, keepdim=True)
        rr, nn = r[XI], n[QI]; dot = (nn * rr).sum(1)
        p1 = (W * 0.5 * (1 + eta * SG * dot)).sum()
        rp = dot[:, None] * nn + s * (rr - dot[:, None] * nn)
        rho = 0.5 * (torch.eye(2, dtype=torch.complex128) + torch.einsum("bk,kij->bij", rp.to(torch.complex128), PAULI))
        p2 = 0.0
        for q in Q2:
            E = povm(B[q])[AI[q]]
            p2 = p2 + (W[GI[q]] * torch.einsum("bij,bji->b", E, rho[GI[q]]).real).sum()
        return p1, p2
    opt = torch.optim.Adam([rv, nv] + list(B.values()), lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
    for _ in range(STEPS):
        p1, p2 = evaluate(); d1, d2 = p1 - corner[0], p2 - corner[1]
        loss = -(0.5 * (d1 + d2) - 0.5 * (d1 - d2).abs()); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        p1, p2 = evaluate()
    return float(p1), float(p2)


def main():
    arms = {"S1_2to1": arm_S1(), "S2_C3": arm_S2(3), "S2_C6": arm_S2(6), "S3_nested": arm_S3()}
    out = {"doc": "100 (8a02346)", "arms": {}}
    for name, arm in arms.items():
        curve = []
        for eta in ETAS:
            best = max((solve(arm, eta, 100 * ETAS.index(eta) + rs) for rs in range(RESTARTS)),
                       key=lambda t: min(t[0] - arm[4][0], t[1] - arm[4][1]))
            curve.append({"eta": eta, "P1": best[0], "P2": best[1], "gain": min(best[0] - arm[4][0], best[1] - arm[4][1])})
        top = max(curve, key=lambda c: c["gain"])
        out["arms"][name] = {"classical_corner": arm[4], "curve": curve, "best": top, "seq_gain": top["gain"] >= 0.02}
        print(name, top, flush=True)
    s1 = out["arms"]["S1_2to1"]["best"]
    out["G_MTB"] = {"symmetric_min": min(s1["P1"], s1["P2"]), "pass": min(s1["P1"], s1["P2"]) >= 0.5 + 0.8 / (2 * math.sqrt(2)) - 1e-6}
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp103.json", out)
    print(json.dumps({"G_MTB": out["G_MTB"], "seq_gain": {k: v["seq_gain"] for k, v in out["arms"].items()}}, indent=1))


if __name__ == "__main__":
    main()
