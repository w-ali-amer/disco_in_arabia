# -*- coding: utf-8 -*-
"""exp133_run.py -- doc 133: white-noise critical visibility of the which-referent two-qubit strategies (qnlp_mac_env).
quantum_capture() is exp100_accounting.quantum() copied verbatim, except that it also returns the best restart's states
and POVMs (exp100_accounting.py is not modified). G-REPRO: regenerated values must equal the stored exp113 values.
usage: python -I exp133_run.py probe   -> exp133_logs/probe.json (determinism probe, step 0)
       python -I exp133_run.py run TOL -> results_exp133.json + exp133_strategies.npz (Rule 3)"""
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp100_accounting as A100, exp110_run as R110, exp94_common as C94

torch.set_default_dtype(torch.float64)
OUT = os.path.join(ROOT, "results_exp133.json"); NPZ = os.path.join(ROOT, "exp133_strategies.npz")
LOGS = os.path.join(ROOT, "exp133_logs")
STORED = {3: 0.7781249235008523, 4: 0.6314295355281454}          # results_exp113.json
P_C = {3: 13 / 18, 4: 812 / 1440}                                 # exp116 / exp132 exact two-bit values
MAX_H = 12.0


def quantum_capture(pb, m, restarts, steps, seed0=0, on_restart=None):
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
    for rs in range(restarts):
        g = torch.Generator().manual_seed(100000 + 1000 * m + rs + seed0)
        psi = torch.randn(2, len(X), D, generator=g).requires_grad_()
        Bs = {r: torch.randn(2, len(A[r]), D, D, generator=g).requires_grad_() for r in R}
        opt = torch.optim.Adam([psi] + list(Bs.values()), lr=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
        for _ in range(steps):
            loss = -success(psi, Bs); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            v = float(success(psi, Bs))
        if v > best[0]:
            best = (v, (psi.detach().clone(), {r: b.detach().clone() for r, b in Bs.items()}))
        if on_restart:
            on_restart(rs)
    psi_ri, Bs = best[1]
    with torch.no_grad():
        ok = True; worst_neg = 0.0; worst_sum = 0.0
        for r in R:
            E = povm(Bs[r]); ev = torch.linalg.eigvalsh(E)
            worst_neg = min(worst_neg, float(ev.min()))
            worst_sum = max(worst_sum, float((E.sum(0) - torch.eye(D, dtype=E.dtype)).abs().max()))
        ok = worst_neg >= -1e-10 and worst_sum <= 1e-10
        val = float(success(psi_ri, Bs))
        # --- added for exp133: the strategy itself and the decoder's success on the maximally mixed state ---
        psi = torch.complex(psi_ri[0], psi_ri[1]); psi = psi / psi.norm(dim=1, keepdim=True)
        Es = {r: povm(Bs[r]) for r in R}
        f_mix = float(sum((W[r] * (torch.einsum("aii->a", Es[r]).real / D)[None, :]).sum() for r in R))
    return {"value": val, "G_POVM": ok, "min_eig": worst_neg, "max_sum_dev": worst_sum, "restarts": restarts, "steps": steps,
            "f_mix": f_mix, "psi": psi.numpy(), "E": {r: Es[r].numpy() for r in R}}


def vstar(pc, pq, f):
    return (pc - f) / (pq - f)


def qrac21_problem():
    rows = [((b0, b1), y, (b0, b1)[y], 1.0) for b0 in (0, 1) for b1 in (0, 1) for y in (0, 1)]
    return A100.problem(rows)


def probe():
    pb = A100.problem(R110.uniform_rows(3)); out = {}
    for tag, th in (("default_a", None), ("default_b", None), ("threads1", 1)):
        if th:
            torch.set_num_threads(th)
        out[tag] = repr(quantum_capture(pb, 2, 2, 150)["value"])
        print(tag, out[tag], flush=True)
    out["default_threads"] = os.cpu_count(); out["bit_exact"] = len({out["default_a"], out["default_b"], out["threads1"]}) == 1
    os.makedirs(LOGS, exist_ok=True); json.dump(out, open(os.path.join(LOGS, "probe.json"), "w"), indent=1)
    print("bit_exact", out["bit_exact"], flush=True)


def run(tol):
    assert not os.path.exists(OUT) and not os.path.exists(NPZ), "Rule 3"
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"doc": "133 (exp133)", "git_head": head, "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "tolerance_G_REPRO": tol, "torch_threads": torch.get_num_threads(), "gates": {}, "cases": {}}
    t0 = time.time(); arrays = {}
    # calibrations (same code path)
    q = quantum_capture(qrac21_problem(), 1, 64, 1500)
    pq_ref = 0.5 * (1 + 1 / math.sqrt(2)); v = vstar(0.75, q["value"], q["f_mix"])
    res["cases"]["qrac21"] = {"P_Q": q["value"], "P_Q_closed": pq_ref, "P_C": 0.75, "f_mix": q["f_mix"], "v_star": v,
                              "v_star_closed": 1 / math.sqrt(2), "G_POVM": q["G_POVM"]}
    res["gates"]["G_CAL1"] = bool(abs(q["value"] - pq_ref) <= 1e-6 and abs(v - 1 / math.sqrt(2)) <= 1e-4 and q["G_POVM"])
    pb2 = A100.problem(R110.uniform_rows(2)); q2 = quantum_capture(pb2, 1, 64, 1500)
    pq2 = 0.5 + (4 * math.sqrt(2) + 2) / 20
    res["cases"]["K2_d2"] = {"P_Q": q2["value"], "P_Q_closed": pq2, "P_C": 0.8, "f_mix": q2["f_mix"], "v_star": vstar(0.8, q2["value"], q2["f_mix"]),
                             "G_POVM": q2["G_POVM"]}
    res["gates"]["G_CAL2"] = bool(abs(q2["value"] - pq2) <= 1e-6 and q2["G_POVM"])
    print("CAL", res["gates"], json.dumps({k: {kk: vv for kk, vv in c.items()} for k, c in res["cases"].items()}), round(time.time() - t0), "s", flush=True)
    if not (res["gates"]["G_CAL1"] and res["gates"]["G_CAL2"]):
        res["stopped"] = "calibration failed; K = 3/4 not read"; C94.write_new_json(OUT, res); print("STOP", flush=True); return
    for K in (3, 4):
        pb = A100.problem(R110.uniform_rows(K)); t1 = time.time()
        def tick(rs, K=K, t1=t1):
            if rs == 7:
                proj = (time.time() - t1) / 8 * 256 / 3600 * (1 if K == 4 else 3)   # K=3 projection covers K=4 (~3x larger)
                print("K", K, "first 8 restarts", round(time.time() - t1), "s; projected", round(proj, 2), "h", flush=True)
                if proj > MAX_H:
                    raise SystemExit("time rule: projected %.2f h > %g h" % (proj, MAX_H))
        q = quantum_capture(pb, 2, 256, 1500, on_restart=tick)
        rep = abs(q["value"] - STORED[K]) <= tol
        res["gates"]["G_REPRO_K%d" % K] = bool(rep); res["gates"]["G_POVM_K%d" % K] = bool(q["G_POVM"])
        res["cases"]["K%d_d4" % K] = {"P_Q": q["value"], "P_Q_stored_exp113": STORED[K], "abs_diff": abs(q["value"] - STORED[K]),
                                      "P_C": P_C[K], "f_mix": q["f_mix"], "f_unif": 1 / K, "wall_s": time.time() - t1,
                                      "min_eig": q["min_eig"], "max_sum_dev": q["max_sum_dev"]}
        print("K", K, res["cases"]["K%d_d4" % K], "G_REPRO", rep, flush=True)
        if not rep:
            res["stopped"] = "G-REPRO failed at K = %d; nothing read" % K; C94.write_new_json(OUT, res); print("STOP", flush=True); return
        c = res["cases"]["K%d_d4" % K]; c["v_star"] = vstar(P_C[K], q["value"], q["f_mix"]); c["noise_tolerance"] = 1 - c["v_star"]
        arrays["K%d_psi" % K] = q["psi"]
        for r, E in q["E"].items():
            arrays["K%d_E_%s" % (K, r)] = E
    res["all_gates_pass"] = all(res["gates"].values()); res["wall_s"] = time.time() - t0
    res["provenance"] = C94.provenance()
    np.savez(NPZ, **arrays); C94.write_new_json(OUT, res)
    print("DONE", res["gates"], {k: c.get("v_star") for k, c in res["cases"].items()}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        probe()
    else:
        run(float(sys.argv[2]))
