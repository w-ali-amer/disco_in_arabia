# -*- coding: utf-8 -*-
"""exp108_run.py -- doc 106 (REGISTERED 8c7e3bd): one-atom DisCoCirc.

(a) Compositional which-referent cap: state = U_{c2} U_{c1} |g> on the Bloch sphere (SO(3) rotations), sharp binary
    measurement along n_v ('+' = first).  Boxes: P1 = one physical pulse (rotation about an xy-plane axis; 2 params),
    SU2 = P1 then virtual Z (3 params).  STRICT (same box per class in both positions) / RELAXED (position-indexed).
    Adam, 256 restarts, float64, exact re-evaluation (lower bounds).  Free code (exp101) re-derived (G-FREE).
    K=3 referents theory arm (free qubit via exp100 optimiser; classical 1 bit exact via MILP).
(b) Story-level pipeline check on exp95 draws e98p/e98c (test_id/oodl/long0): fresh index copy per event, router
    success at the question event; efficiency table.  Writes results_exp108.json (Rule 3).
"""
import itertools, json, math
import numpy as np
import torch
import exp94_common as C94
import exp95_data as D95
import exp100_accounting as A100
import exp101_game as G101

torch.set_default_dtype(torch.float64)
RESTARTS, STEPS = 256, 1500


def rot(axis, ang):
    """Rodrigues rotation matrices, batched: axis (...,3) unit, ang (...)."""
    K = torch.zeros(axis.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    K = K - K.transpose(-1, -2)
    I = torch.eye(3).expand_as(K)
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    return I + s * K + (1 - c) * (K @ K)


def box(params, kind):
    """params (..., 3): theta, phi, alpha. P1: rotation by theta about (cos phi, sin phi, 0). SU2: then Rz(alpha)."""
    th, ph, al = params[..., 0], params[..., 1], params[..., 2]
    ax = torch.stack([torch.cos(ph), torch.sin(ph), torch.zeros_like(ph)], -1)
    R = rot(ax, th)
    if kind == "SU2":
        R = rot(torch.tensor([0.0, 0.0, 1.0]).expand_as(ax), al) @ R
    return R


def comp_success(C, kind, strict, seed):
    g = torch.Generator().manual_seed(108000 + seed)
    P = torch.randn(C if strict else 2 * C, 3, generator=g).requires_grad_()
    N = torch.randn(C, 3, generator=g).requires_grad_()
    pairs = torch.tensor(list(itertools.permutations(range(C), 2)))
    z = torch.tensor([0.0, 0.0, 1.0])
    def succ():
        B = box(P, kind)
        B1 = B[pairs[:, 0]]; B2 = B[pairs[:, 1]] if strict else B[C + pairs[:, 1]]
        r = (B2 @ (B1 @ z[:, None]))[..., 0]
        n = N / N.norm(dim=1, keepdim=True)
        return (0.5 + 0.25 * ((n[pairs[:, 0]] * r).sum(1) - (n[pairs[:, 1]] * r).sum(1))).mean()
    opt = torch.optim.Adam([P, N], lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
    for _ in range(STEPS):
        loss = -succ(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        return float(succ()), P.detach().tolist(), (N / N.norm(dim=1, keepdim=True)).detach().tolist()


def _job(args):
    torch.set_num_threads(1)
    return comp_success(*args)


def part_a():
    from multiprocessing import Pool
    out = {}
    with Pool(8) as pool:
      for C in (3, 4, 6):
        rec = {"free_exp101": G101.analytic(C)["qubit"], "classical_1bit": G101.analytic(C)["classical"]}
        for kind in ("P1", "SU2"):
            for strict in (True, False):
                jobs = [(C, kind, strict, 1000 * C + 10 * (kind == "SU2") + (not strict) * 5 + 100 * s) for s in range(RESTARTS)]
                best = max(pool.map(_job, jobs), key=lambda t: t[0])
                rec["%s_%s" % ("STRICT" if strict else "RELAXED", kind)] = {"value": best[0], "boxes": best[1], "axes": best[2]}
                print("C", C, kind, "STRICT" if strict else "RELAXED", round(best[0], 6), flush=True)
        out[C] = rec
    return out


def part_a_free_check():
    ok = True; vals = {}
    for C in (3, 4, 6):
        cl = list(range(C)); pb = A100.problem(G101.game_rows(cl, {c: 1.0 for c in cl}))
        q = A100.quantum(pb, 1)["value"]; vals[C] = q; ok &= abs(q - G101.analytic(C)["qubit"]) < 1e-6
    return {"values": vals, "pass": bool(ok)}


def part_k3():
    C = 6; rows = []
    trip = list(itertools.permutations(range(C), 3)); w = 1.0 / (len(trip) * 3)
    for t in trip:
        for pos, v in enumerate(t):
            rows.append((t, v, pos, w))
    pb = A100.problem(rows)
    cl = A100.classical_milp(pb, 1, 1800); q = A100.quantum(pb, 1)
    return {"classical_1bit_exact": cl["value"], "classical_proved_optimal": cl["proved_optimal"],
            "classical_dual_bound": cl["dual_bound"], "qubit_free": q["value"], "G_POVM": q["G_POVM"],
            "margin": q["value"] - (cl["value"] if cl["proved_optimal"] else cl["dual_bound"])}


def story_router(draw, codes):
    """per-story P(correct) = router success at the question event; fresh copy per event (depth-flat by construction)."""
    d = D95.build(draw); CL = [tuple(c.split("-")) for c in []]
    res = {}
    for split in ("test_id", "test_oodl", "test_long0"):
        acc = {k: [] for k in codes}
        for st in d[split]:
            cls = st["meta"]["classes"]; q = st["meta"]["q"]; ag = st["gold_agents"][q]
            c1, c2 = [C94.CLASSES.index(tuple(c.split("-"))) for c in cls]
            v = (c1, c2)[ag]
            for name, (states, axes) in codes.items():
                r = np.array(states[(c1, c2)]); n = np.array(axes[v])
                p_first = 0.5 * (1 + n @ r)
                acc[name].append(p_first if ag == 0 else 1 - p_first)
        res[split] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v))), "n": len(v)}
                      for k, v in acc.items()}
    return res


def free_code():
    n = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    states = {(a, b): list((n[a] - n[b]) / np.linalg.norm(n[a] - n[b])) for a, b in itertools.permutations(range(6), 2)}
    return states, {v: list(n[v]) for v in range(6)}


def comp_code(rec, kind):
    P = torch.tensor(rec["boxes"]); B = box(P, kind); z = torch.tensor([0.0, 0.0, 1.0])
    states = {(a, b): (B[b] @ (B[a] @ z)).tolist() for a, b in itertools.permutations(range(6), 2)}
    return states, {v: rec["axes"][v] for v in range(6)}


def main():
    out = {"doc": "106 (8c7e3bd)"}
    out["G_FREE"] = part_a_free_check(); print("G-FREE", out["G_FREE"], flush=True)
    out["a"] = part_a()
    out["K3"] = part_k3(); print("K3", out["K3"], flush=True)
    a6 = out["a"][6]
    best_kind = next((k for k in ("P1", "SU2") if a6["STRICT_%s" % k]["value"] - a6["classical_1bit"] >= 0.05), None)
    out["one_atom_discocirc"] = {"value": best_kind is not None, "variant": best_kind,
                                 "strict_P1_margin": a6["STRICT_P1"]["value"] - a6["classical_1bit"],
                                 "strict_SU2_margin": a6["STRICT_SU2"]["value"] - a6["classical_1bit"]}
    print("one_atom_discocirc", out["one_atom_discocirc"], flush=True)
    codes = {"QRAC_free": free_code()}
    if best_kind:
        codes["QRAC_comp_%s" % best_kind] = comp_code(a6["STRICT_%s" % best_kind], best_kind)
    else:
        codes["QRAC_comp_SU2"] = comp_code(a6["STRICT_SU2"], "SU2")
    out["b"] = {dr: story_router(dr, codes) for dr in ("e98p", "e98c")}
    gp = True
    for dr in ("e98p", "e98c"):
        for split, r in out["b"][dr].items():
            for name, s in r.items():
                target = 0.5 + (2 * math.sqrt(2) + 1) / 10 if name == "QRAC_free" else a6["STRICT_%s" % name.split("_")[-1]]["value"]
                gp &= abs(s["mean"] - target) <= 3 * s["se"] + 1e-12
    out["G_PIPE"] = bool(gp)
    r98 = json.load(open("results_exp98.json"))
    out["efficiency_table_inputs"] = {"exp98_table": r98["table"],
                                      "Q-AGR-REG_n_params": 1052, "Q-GOLD-REG_n_params": 1024, "learned_control_params": 28,
                                      "TF_d64_n_params": 70081, "classical_1bit_router": 0.8, "classical_2bit_router": 1.0}
    out["router_matches_learning_note"] = "compare QRAC_free test_id mean to exp98 Q-AGR-REG ID (0.874 / 0.897) in the report"
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp108.json", out)
    print(json.dumps({"G_FREE": out["G_FREE"]["pass"], "G_PIPE": out["G_PIPE"], "b": out["b"]}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
