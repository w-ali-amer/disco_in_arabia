# -*- coding: utf-8 -*-
"""exp137_e3c.py -- doc 137 E3c (REGISTERED ea85cbd): is the K = 3 dip real?
  s1     (qnlp_mac_env): exp100 quantum() copy that keeps the best strategy; middle priors x seed0 {256, 512, 768}, U and E-6 at seed0 0
  s2     (rac_env):      RAC-tools seesaw at I_0.25 and I_0.75 (prior from E3C_PRIOR), as E3a e3r
  s3     (qnlp_mac_env): cross-evaluate every saved strategy under the five priors, gates, prediction -> results_exp137_E3c.json (Rule 3)
"""
import hashlib, json, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp137_run as R137
import exp137_e3 as E3

LOGS = R137.LOGS; OUT = os.path.join(ROOT, "results_exp137_E3c.json"); STRAT = os.path.join(LOGS, "E3c_strat")
FIVE = ["U", "I_0.25", "I_0.5", "I_0.75", "E-6"]; MID = ["I_0.25", "I_0.5", "I_0.75"]; RESTARTS = 256


def part_path(name):
    return os.path.join(LOGS, "E3c_%s.json" % name)


def write_part(name, obj):
    with open(part_path(name), "x") as f:
        json.dump(obj, f, indent=1)


def tensors(pb, m):
    import torch
    torch.set_default_dtype(torch.float64)
    D = 2 ** m; X, Rr, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    xi = {x: i for i, x in enumerate(X)}
    W = {r: torch.zeros(len(X), len(A[r])) for r in Rr}
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
        for r in Rr:
            E = povm(Bs[r])
            pr = torch.einsum("xi,aij,xj->xa", psi.conj(), E, psi).real
            tot = tot + (W[r] * pr).sum()
        return tot
    return D, X, Rr, A, povm, success


def quantum_keep(pb, m, seed0):
    """exp100_accounting.quantum, line for line (STEPS 1500, Adam lr 0.05, cosine), plus the best strategy."""
    import torch
    D, X, Rr, A, povm, success = tensors(pb, m); STEPS = 1500
    best = (-1, None)
    for rs in range(RESTARTS):
        g = torch.Generator().manual_seed(100000 + 1000 * m + rs + seed0)
        psi = torch.randn(2, len(X), D, generator=g).requires_grad_()
        Bs = {r: torch.randn(2, len(A[r]), D, D, generator=g).requires_grad_() for r in Rr}
        opt = torch.optim.Adam([psi] + list(Bs.values()), lr=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        for _ in range(STEPS):
            loss = -success(psi, Bs); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            v = float(success(psi, Bs))
        if v > best[0]:
            best = (v, (psi.detach().clone(), {r: b.detach().clone() for r, b in Bs.items()}))
    return best[1]


def check(pb, strat, m=2):
    import torch
    D, X, Rr, A, povm, success = tensors(pb, m); psi_ri, Bs = strat
    with torch.no_grad():
        worst_neg = 0.0; worst_sum = 0.0
        for r in Rr:
            E = povm(Bs[r]); worst_neg = min(worst_neg, float(torch.linalg.eigvalsh(E).min()))
            worst_sum = max(worst_sum, float((E.sum(0) - torch.eye(D, dtype=E.dtype)).abs().max()))
        return float(success(psi_ri, Bs)), worst_neg >= -1e-10 and worst_sum <= 1e-10


def s1_job(item):
    import torch, exp100_accounting as A100
    torch.set_num_threads(1)
    k, seed0 = item; p = list(R137.priors()[k]); pb = A100.problem(E3.rows(p)); t0 = time.time()
    strat = quantum_keep(pb, 2, seed0); v, ok = check(pb, strat)
    torch.save(strat, os.path.join(STRAT, "%s_s%d.pt" % (k, seed0)))
    return {"prior": k, "seed0": seed0, "value": v, "G_POVM": ok, "wall_s": time.time() - t0}


def s1():
    from multiprocessing import Pool
    os.makedirs(STRAT)
    jobs = [(k, s) for k in MID for s in (256, 512, 768)] + [("U", 0), ("E-6", 0)]
    out = []
    with Pool(9) as pool:
        for r in pool.imap_unordered(s1_job, jobs):
            out.append(r); print(json.dumps(r), flush=True)
    write_part("s1", out)


def s2():
    import itertools, math
    import numpy as np
    import exp136_ractools_check as A                                            # applies the MOSEK -> Clarabel redirect
    k = os.environ["E3C_PRIOR"]; U = A.U; p = list(R137.priors()[k]); N, M, D, seeds = 6, 4, 4, 60
    valid = {}
    for t in E3.X:
        x = [0] * N
        for pos, v in enumerate(t):
            x[v] = pos + 1
        valid[tuple(x)] = math.prod(p[v] for v in t)
    tot = sum(valid.values()); T = {}
    for key in itertools.product(itertools.product(range(M), repeat=N), range(N), range(M)):
        x, y, b = key; w = valid.get(x, 0.0)
        T[key] = w / tot / E3.K if (w and x[y] != 0 and b == x[y]) else 0.0
    assert abs(sum(T.values()) - 1) < 1e-12
    t0 = time.time(); np.random.seed(13730)
    r = U.perform_seesaw(N, D, seeds, m=M, bias_tensor=T, verbose=False, mosek_accuracy=1e-10)
    out = {"prior": k, "seesaw": float(r["optimal value"]), "seeds": seeds, "wall_s": time.time() - t0}
    print(json.dumps(out), flush=True); write_part("s2_%s" % k, out)


def s3():
    import torch, exp100_accounting as A100
    assert not os.path.exists(OUT), "Rule 3"
    e3a = json.load(open(E3.OUT))["rows"]; s1r = json.load(open(part_path("s1")))
    s2r = {k: json.load(open(part_path("s2_%s" % k))) for k in ("I_0.25", "I_0.75")}
    pbs = {k: A100.problem(E3.rows(list(R137.priors()[k]))) for k in FIVE}
    cross = {}; povm_ok = True
    for fn in sorted(os.listdir(STRAT)):
        strat = torch.load(os.path.join(STRAT, fn))
        for k in FIVE:
            v, ok = check(pbs[k], strat); povm_ok &= ok; cross["%s|%s" % (fn[:-3], k)] = v
    rows = {}
    for k in FIVE:
        cands = {"e3a": e3a[k]["lower"]}
        cands.update({"strat:" + key.split("|")[0]: v for key, v in cross.items() if key.endswith("|" + k)})
        if k in s2r:
            cands["ractools"] = s2r[k]["seesaw"]
        src = max(cands, key=cands.get); lo = cands[src]
        rows[k] = {"classical": e3a[k]["classical"], "upper": e3a[k]["upper"], "lower_e3a": e3a[k]["lower"], "lower_new": lo, "source": src,
                   "margin_e3a": e3a[k]["lower"] - e3a[k]["classical"], "margin_new": lo - e3a[k]["classical"], "candidates": cands}
    g = {"G_POVM": bool(povm_ok and all(r["G_POVM"] for r in s1r)),
         "G_REPRO_STRAT": all(abs(r["value"] - e3a[r["prior"]]["lower_exp100"]) <= 1e-6 for r in s1r if r["seed0"] == 0),
         "G_SANDWICH_C": all(r["lower_new"] <= r["upper"] + 1e-6 for r in rows.values())}
    m = {k: rows[k]["margin_new"] for k in FIVE}
    pred = {"P3c_dip": m["I_0.5"] < min(m["U"], m["E-6"]) - 0.002, "margins_new": m}
    res = {"doc": "137 E3c", "registered": "ea85cbd", "gates": g, "prediction": pred, "rows": rows, "cross": cross, "s1": s1r, "s2": s2r,
           "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"gates": g, "prediction": pred}, indent=1))
    for k, r in rows.items():
        print("%-7s classical %.6f lower %.6f -> %.6f (%s) margin %.4f -> %.4f upper %.6f" % (k, r["classical"], r["lower_e3a"], r["lower_new"], r["source"], r["margin_e3a"], r["margin_new"], r["upper"]))


if __name__ == "__main__":
    {"s1": s1, "s2": s2, "s3": s3}[sys.argv[1]]()
