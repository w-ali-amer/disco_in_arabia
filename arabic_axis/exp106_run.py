# -*- coding: utf-8 -*-
"""exp106_run.py -- doc 104 (REGISTERED 8c7e3bd; A1 04d163c; lexicon 9ed7d59): SRE-QRAC template x field.

x = (t, f) in 4 x 4; question q in {TEMPLATE, FIELD} uniform; 4-way answer; 2 qubits (D = 4) vs 2 bits.
Classical EXACT (brute force over decoders, m = 2); quantum by exp100's general optimiser (D = 4); C-COMP descriptive
(state = U_f U_t |0>, one U(4) per class, fixed order, + 4-outcome POVM per question; 64 restarts; lower bound).
Arms: U (uniform; G-THEORY 0.75 / 0.625), E (PADT joint; primary gate >= 0.05), E-noweak (sensitivity, descriptive).
Writes results_exp106.json (Rule 3).
"""
import itertools, json, math
import torch
import exp94_common as C94
import exp100_accounting as A100

torch.set_default_dtype(torch.float64)
T = ("AGENT", "PATIENT", "PLACE", "MASDAR"); F = ("POL", "ECON", "SEC", "SOC")


def rows_for(p):
    rows = []
    for (t, f), w in p.items():
        rows.append(((t, f), "TEMPLATE", t, w / 2)); rows.append(((t, f), "FIELD", f, w / 2))
    return rows


def ahu(G):
    Gc = torch.complex(G[..., 0, :, :], G[..., 1, :, :])
    return torch.matrix_exp(Gc - Gc.conj().transpose(-1, -2))


def comp(p, restarts=64, steps=1500):
    X = list(p); W = torch.tensor([p[x] for x in X]); W = W / W.sum()
    ti = torch.tensor([T.index(t) for t, _ in X]); fi = torch.tensor([F.index(f) for _, f in X])
    best = -1.0
    for rs in range(restarts):
        g = torch.Generator().manual_seed(106000 + rs)
        Gt = (0.5 * torch.randn(4, 2, 4, 4, generator=g)).requires_grad_(); Gf = (0.5 * torch.randn(4, 2, 4, 4, generator=g)).requires_grad_()
        Bq = {q: torch.randn(2, 4, 4, 4, generator=g).requires_grad_() for q in ("T", "F")}
        def povm(B):
            Bc = torch.complex(B[0], B[1]); Gm = Bc @ Bc.conj().transpose(-1, -2)
            ev, U = torch.linalg.eigh(Gm.sum(0)); Sm = U @ torch.diag(ev.clamp(min=1e-12) ** -0.5).to(U.dtype) @ U.conj().T
            return Sm @ Gm @ Sm
        def succ():
            e0 = torch.zeros(4, dtype=torch.complex128); e0[0] = 1
            psi = (ahu(Gf)[fi] @ (ahu(Gt)[ti] @ e0[:, None]))[..., 0]
            Et, Ef = povm(Bq["T"]), povm(Bq["F"])
            pt = torch.einsum("xi,xij,xj->x", psi.conj(), Et[ti], psi).real
            pf = torch.einsum("xi,xij,xj->x", psi.conj(), Ef[fi], psi).real
            return (W * 0.5 * (pt + pf)).sum()
        opt = torch.optim.Adam([Gt, Gf] + list(Bq.values()), lr=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
        for _ in range(steps):
            loss = -succ(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            best = max(best, float(succ()))
    return best


def arm(p):
    pb = A100.problem(rows_for(p))
    cl = A100.classical_brute(pb, 2); q = A100.quantum(pb, 2)
    return {"classical_exact": cl, "quantum": q["value"], "G_POVM": q["G_POVM"], "margin": q["value"] - cl,
            "comp_descriptive": comp(p)}


def main():
    j = json.load(open("joint_exp106.json"))
    uni = {(t, f): 1.0 for t in T for f in F}
    emp = {tuple(k.split("|")): v for k, v in j["joint"].items()}
    emp_nw = {tuple(k.split("|")): v for k, v in j["joint_noweak"].items()}
    out = {"doc": "104 (8c7e3bd; A1 04d163c; lexicon 9ed7d59)", "joint_counts": j["joint"], "tokens_in_joint": j["tokens_in_joint"]}
    out["U"] = arm(uni); print("U", out["U"], flush=True)
    out["G_THEORY"] = {"pass": bool(abs(out["U"]["classical_exact"] - 0.625) < 1e-6 and abs(out["U"]["quantum"] - 0.75) < 1e-6)}
    out["E"] = arm(emp); print("E", out["E"], flush=True)
    out["E_noweak"] = arm(emp_nw); print("E_noweak", out["E_noweak"], flush=True)
    out["sre_qrac_empirical"] = bool(out["E"]["margin"] >= 0.05)
    out["split_descriptive"] = j["split_descriptive"]
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp106.json", out)
    print(json.dumps({"G_THEORY": out["G_THEORY"], "sre_qrac_empirical": out["sre_qrac_empirical"]}), flush=True)


if __name__ == "__main__":
    main()
