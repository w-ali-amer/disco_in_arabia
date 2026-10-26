# -*- coding: utf-8 -*-
"""exp132_run.py -- doc 132: exact two-bit value of the K = 4 which-referent game by exhaustive enumeration (qnlp_mac_env).
Builds exp132_enum.c, runs G-CAL, G-SYM3, G-COVER, G-EVAL on K <= 3, then the timed K = 4 run, then G-EVAL/G-BRACKET and the reading.
usage: python -I exp132_run.py -> results_exp132.json (Rule 3)."""
import hashlib, json, math, os, subprocess, sys, time
from fractions import Fraction
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp100_accounting as A100, exp110_run as R110

SRC, BIN = os.path.join(ROOT, "exp132_enum.c"), os.path.join(ROOT, "exp132_enum")
OUT = os.path.join(ROOT, "results_exp132.json")
POOL, STRIDE, MAX_H = 9, 8, 12.0
Q_LB, Q_UB, C_UB_118, INC_118 = 0.6314295355281454, 0.6607919383929826, 0.7291666666666666, 812


def run_job(a):
    K, C, M, mode, lo, hi, stride, j = a
    o = subprocess.run([BIN] + [str(v) for v in a], capture_output=True, text=True, check=True).stdout.split()
    v = list(map(int, o)); npar = 2 if M == 2 else 4
    return {"args": list(a), "T": v[0], "arg": v[1:1 + npar], "count": v[1 + npar], "nrep": v[2 + npar], "N": v[3 + npar], "L": v[4 + npar]}


def nrep_of(K, C, mode):
    return K ** C if mode == 1 else math.comb(C + K - 1, K - 1)


def search(pool, K, C, M, mode, stride=STRIDE):
    nr = nrep_of(K, C, mode)
    jobs = [(K, C, M, mode, r, r + 1, stride, j) for r in range(nr) for j in range(stride)]
    rs = pool.map(run_job, jobs)
    return combine(K, C, M, mode, rs)


def combine(K, C, M, mode, rs):
    L, N = rs[0]["L"], rs[0]["N"]; nr = nrep_of(K, C, mode)
    best = max(enumerate(rs), key=lambda t: (t[1]["T"], -t[0]))[1]
    cnt = sum(r["count"] for r in rs)
    if M == 2:
        exp = nr * L if mode == 0 else math.comb(L + 1, 2)
    else:
        exp = nr * math.comb(L + 2, 3) if mode == 0 else math.comb(L + 3, 4)
    assert rs[0]["nrep"] == nr
    return {"K": K, "C": C, "M": M, "mode": mode, "T": best["T"], "value": best["T"] / (N * K), "argmax": best["arg"], "N": N, "L": L,
            "count": cnt, "count_expected": exp, "G_COVER": cnt == exp}


def py_value(K, labs):
    """G-EVAL: score the labellings against exp100's problem weights with the pointwise-optimal encoder."""
    pb = A100.problem(R110.uniform_rows(K)); ci = {c: i for i, c in enumerate(R110.CLASSES)}
    dig = lambda p, v: (p // K ** v) % K
    by_x = {}
    for (x, r, a), w in pb["P"].items():
        by_x.setdefault(x, []).append((r, a, w))
    return sum(max(sum(w for r, a, w in rows if dig(p, ci[r]) == a) for p in labs) for rows in by_x.values())


def main():
    assert not os.path.exists(OUT), "Rule 3"
    os.makedirs(os.path.join(ROOT, "exp132_logs"), exist_ok=True)
    subprocess.run(["clang", "-O3", "-mcpu=native", "-o", BIN, SRC], check=True)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"doc": "132 (exp132)", "git_head": head, "src_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
           "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "gates": {}}
    t0 = time.time()
    with Pool(POOL) as pool:
        # G-CAL
        cal = []
        for C in range(2, 7):
            r = search(pool, 2, C, 2, 0, stride=1); f = 0.5 + (C // 2) * ((C + 1) // 2) / (C * (C - 1))
            r.update(known=f, ok=abs(r["value"] - f) <= 1e-12); cal.append(r)
            r = search(pool, 2, C, 4, 0, stride=1); r.update(known=1.0, ok=r["T"] == r["N"] * 2); cal.append(r)
        k3 = search(pool, 3, 6, 4, 0); k3.update(known=13 / 18, ok=Fraction(k3["T"], 360) == Fraction(13, 18)); cal.append(k3)
        res["calibrations"] = cal
        res["gates"]["G_CAL"] = all(c["ok"] for c in cal)
        print("G-CAL", res["gates"]["G_CAL"], [(c["K"], c["C"], c["M"], round(c["value"], 6), c["ok"]) for c in cal], round(time.time() - t0), "s", flush=True)
        k3u = search(pool, 3, 6, 4, 1); res["k3_unreduced"] = k3u
        res["gates"]["G_SYM3"] = k3u["T"] == k3["T"]
        print("G-SYM3", res["gates"]["G_SYM3"], k3["T"], k3u["T"], round(time.time() - t0), "s", flush=True)
        res["gates"]["G_COVER_small"] = all(c["G_COVER"] for c in cal) and k3u["G_COVER"]
        e3 = py_value(3, k3["argmax"]); res["gates"]["G_EVAL_K3"] = abs(e3 - k3["value"]) <= 1e-12; res["k3_py_value"] = e3
        print("G-COVER", res["gates"]["G_COVER_small"], "G-EVAL K3", res["gates"]["G_EVAL_K3"], e3, flush=True)
        if not all(res["gates"].values()):
            res["stopped"] = "small-case gate failed; K = 4 not run"; json.dump(res, open(OUT, "w"), indent=1); print("STOP", flush=True); return
        # K = 4, timed first batch of POOL concurrent jobs
        nr = nrep_of(4, 6, 0); jobs = [(4, 6, 4, 0, r, r + 1, STRIDE, j) for r in range(nr) for j in range(STRIDE)]
        t1 = time.time(); first = pool.map(run_job, jobs[:POOL]); tb = time.time() - t1
        proj_h = tb * len(jobs) / POOL / 3600; res["k4_pilot"] = {"batch_s": tb, "jobs": len(jobs), "projected_h": proj_h}
        print("K4 pilot batch", round(tb, 1), "s; projected", round(proj_h, 2), "h", flush=True)
        if proj_h > MAX_H:
            res["stopped"] = "time rule: projected %.2f h > %g h" % (proj_h, MAX_H); json.dump(res, open(OUT, "w"), indent=1); print("STOP", flush=True); return
        rest = pool.map(run_job, jobs[POOL:]); k4 = combine(4, 6, 4, 0, first + rest)
        res["k4"] = k4; res["k4_wall_s"] = time.time() - t1
    res["gates"]["G_COVER_K4"] = k4["G_COVER"]
    e4 = py_value(4, k4["argmax"]); res["k4_py_value"] = e4; res["gates"]["G_EVAL_K4"] = abs(e4 - k4["value"]) <= 1e-12
    T = k4["T"]; res["gates"]["G_BRACKET"] = INC_118 <= T <= round(C_UB_118 * 1440)
    res["all_gates_pass"] = all(res["gates"].values())
    if res["all_gates_pass"]:
        V = T / 1440
        if Q_LB - V >= 0.05: rd = "TRUE"
        elif Q_UB - V >= 0.05: rd = "UNDETERMINED"
        else: rd = "FALSE"
        res["reading"] = {"T": T, "V": V, "multi_referent_2q": rd, "margin_vs_quantum_LB": Q_LB - V, "max_margin_vs_quantum_UB": Q_UB - V,
                          "strict_advantage_certified": V < Q_LB, "two_bits_beat_two_qubits": V > Q_UB, "prediction_T812": T == 812}
    res["wall_s"] = time.time() - t0
    print("K4", T, k4["value"], "gates", res["gates"], "reading", res.get("reading"), flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("DONE", flush=True)


if __name__ == "__main__":
    main()
