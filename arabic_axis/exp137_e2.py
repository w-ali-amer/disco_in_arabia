# -*- coding: utf-8 -*-
"""exp137_e2.py -- doc 137 E2 (amendment REGISTERED 3b3aed3).  Run with qnlp_mac_env (torch + scipy).
  e2i   weighted unreduced enumeration (exp137_enum.c): G-ENUM-K2 (K = 2, M = 2 vs E1 exact classical, all nine priors),
        G-ENUM-U3 (uniform K = 3 = 13/18), then K = 3, M = 4 for all nine priors; G-EVAL Python re-score of every argmax.
  e2ii  exp116 A1 (reps + fibre-lex rows), weighted objective, E-6, K = 3; 15 reps on Pool(9), 4 h each (<= 8 h wall).
  e2iii K = 4 cost pilot at E-6: one Pool batch of mid-range jobs, throughput -> projected full time.
  e2    combine parts, gates -> results_exp137_E2.json (Rule 3).
"""
import hashlib, itertools, json, math, os, subprocess, sys, time
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp137_run as R137

LOGS = R137.LOGS; OUT = os.path.join(ROOT, "results_exp137_E2.json")
SRC, BIN = os.path.join(ROOT, "exp137_enum.c"), os.path.join(ROOT, "exp137_enum")
POOL = 9; C6 = 6


def part_path(name):
    return os.path.join(LOGS, "E2_%s.json" % name)


def write_part(name, obj):
    os.makedirs(LOGS, exist_ok=True)
    with open(part_path(name), "x") as f:
        json.dump(obj, f, indent=1)


def prior_file(name, p):
    path = os.path.join(LOGS, "prior_%s.txt" % name)
    with open(path, "w") as f:
        f.write(" ".join("%.17g" % v for v in p) + "\n")
    return path


def run_job(a):
    o = subprocess.run([BIN] + [str(v) for v in a], capture_output=True, text=True, check=True).stdout.split()
    M = a[2]; npar = 2 if M == 2 else 4
    return {"args": list(a), "V": float(o[0]), "arg": list(map(int, o[1:1 + npar])), "count": int(o[1 + npar]), "N": int(o[2 + npar]), "L": int(o[3 + npar])}


def search(pool, K, M, pf):
    L = K ** C6
    rs = pool.map(run_job, [(K, C6, M, p0, p0 + 1, 1, 0, pf) for p0 in range(L)])
    best = max(enumerate(rs), key=lambda t: (t[1]["V"], -t[0]))[1]; cnt = sum(r["count"] for r in rs)
    return {"K": K, "M": M, "value": best["V"], "argmax": best["arg"], "count": cnt, "count_expected": math.comb(L + M - 1, M),
            "G_COVER": cnt == math.comb(L + M - 1, M)}


def rows(K, p):
    out = []
    for t in itertools.permutations(range(C6), K):
        w = math.prod(p[v] for v in t)
        for pos, v in enumerate(t):
            out.append((t, v, pos, w / K))
    return out


def py_value(K, p, labs):
    import exp100_accounting as A100
    pb = A100.problem(rows(K, p)); dig = lambda q, v: (q // K ** v) % K; by_x = {}
    for (x, r, a), w in pb["P"].items():
        by_x.setdefault(x, []).append((r, a, w))
    return sum(max(sum(w for r, a, w in rs if dig(q, r) == a) for q in labs) for rs in by_x.values())


def e2i():
    subprocess.run(["clang", "-O3", "-mcpu=native", "-o", BIN, SRC], check=True)
    P = R137.priors(); e1 = json.load(open(R137.part_path("q"))); out = {"src_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(), "k2": {}, "k3": {}}
    t0 = time.time()
    with Pool(POOL) as pool:
        for k, p in P.items():
            r = search(pool, 2, 2, prior_file(k, p)); r["e1_classical"] = e1[k]["classical"]; r["diff"] = abs(r["value"] - e1[k]["classical"])
            r["py_value"] = py_value(2, list(p), r["argmax"]); out["k2"][k] = r
        print("K2", {k: (round(v["value"], 9), v["diff"], v["G_COVER"]) for k, v in out["k2"].items()}, round(time.time() - t0), "s", flush=True)
        for k, p in P.items():
            t1 = time.time(); r = search(pool, 3, 4, prior_file(k, p)); r["wall_s"] = time.time() - t1
            r["py_value"] = py_value(3, list(p), r["argmax"]); out["k3"][k] = r
            print("K3", k, r["value"], r["argmax"], r["G_COVER"], abs(r["py_value"] - r["value"]), round(r["wall_s"]), "s", flush=True)
    out["wall_s"] = time.time() - t0; write_part("i", out)


def milp_job(args):
    import exp110_run as R110, exp116_run as E116
    p = R137.priors()["E-6"]; R110.uniform_rows = lambda K: rows(K, list(p))
    return E116.solve(args)


def milp_job_scaled(args):
    import exp116_run as E116
    orig = getattr(E116, "_milp_unscaled", E116.milp); E116._milp_unscaled = orig   # idempotent: Pool workers reuse the module
    E116.milp = lambda c, **kw: orig(c * SCALE, **kw)
    r = milp_job(args)
    for k in ("value", "dual"):
        r[k] = r[k] / SCALE if r[k] is not None else None
    return r


SCALE = 1e4


def e2iis():
    """G-MILP-scaled (doc 137, after the G-MILP failure): same 15 reps, objective x 1e4 to defeat HiGHS's 1e-6 absolute gap."""
    jobs = [("A1ws", 3, r, 14400, True) for r in itertools.combinations_with_replacement(range(3), 4)]
    out = []; t0 = time.time()
    with Pool(POOL) as pool:
        for res in pool.imap_unordered(milp_job_scaled, jobs):
            out.append(res); print(res["rep"], res["value"], res["dual"], res["proved"], int(res["wall"]), int(time.time() - t0), flush=True)
    enum = json.load(open(part_path("i")))["k3"]["E-6"]["value"]; best = max(s["value"] for s in out if s["value"] is not None)
    allp = all(s["proved"] for s in out)
    gate = allp and all(s["value"] <= enum + 1e-9 for s in out) and abs(best - enum) <= 1e-9
    write_part(os.environ.get("E2_IIS_PART", "iis"), {"prior": "E-6", "scale": SCALE, "subs": out, "best": best, "enum": enum, "diff": best - enum, "all_proved": allp,
                       "G_MILP_scaled": gate, "wall_s": time.time() - t0})
    print("G_MILP_scaled", gate, best, enum, best - enum, flush=True)


def e2ii():
    jobs = [("A1w", 3, r, 14400, True) for r in itertools.combinations_with_replacement(range(3), 4)]
    out = []; t0 = time.time()
    with Pool(POOL) as pool:
        for res in pool.imap_unordered(milp_job, jobs):
            out.append(res); print(res["rep"], res["value"], res["dual"], res["proved"], int(res["wall"]), int(time.time() - t0), flush=True)
    write_part("ii", {"prior": "E-6", "subs": out, "wall_s": time.time() - t0})


def e2iii():
    subprocess.run(["clang", "-O3", "-mcpu=native", "-o", BIN, SRC], check=True)
    pf = prior_file("E-6", R137.priors()["E-6"]); L = 4 ** C6
    jobs = [(4, C6, 4, 2048, 2049, 8, j, pf) for j in range(8)] + [(4, C6, 4, 2049, 2050, 8, 0, pf)]
    t0 = time.time()
    with Pool(POOL) as pool:
        rs = pool.map(run_job, jobs)
    wall = time.time() - t0; cnt = sum(r["count"] for r in rs); total = math.comb(L + 3, 4)
    proj_h = total / (cnt / wall) / 3600
    out = {"prior": "E-6", "jobs": [r["args"][:7] for r in rs], "batch_count": cnt, "batch_wall_s": wall, "throughput_per_s": cnt / wall,
           "total_combinations": total, "projected_h": proj_h, "within_24h": proj_h <= 24}
    print(json.dumps(out), flush=True); write_part("iii", out)


def e2():
    assert not os.path.exists(OUT), "Rule 3"
    i, ii, iii = (json.load(open(part_path(n))) for n in ("i", "ii", "iii")); g = {}
    g["G_ENUM_K2"] = all(v["diff"] <= 1e-12 and v["G_COVER"] for v in i["k2"].values())
    g["G_ENUM_U3"] = abs(i["k3"]["U"]["value"] - 13 / 18) <= 1e-12
    g["G_EVAL"] = all(abs(v["py_value"] - v["value"]) <= 1e-12 for d in (i["k2"], i["k3"]) for v in d.values())
    g["G_COVER"] = all(v["G_COVER"] for v in i["k3"].values())
    enum = i["k3"]["E-6"]["value"]; subs = ii["subs"]; proved = [s for s in subs if s["proved"]]
    g["G_MILP"] = all(s["value"] <= enum + 1e-9 for s in proved) and (len(proved) < len(subs) or abs(max(s["value"] for s in proved) - enum) <= 1e-9)
    milp = {"n_reps": len(subs), "n_proved": len(proved), "best_incumbent": max(s["value"] for s in subs if s["value"] is not None),
            "certified_upper": max(s["value"] if s["proved"] else s["dual"] for s in subs), "wall_s": ii["wall_s"]}
    res = {"doc": "137 E2", "registered": "3b3aed3", "gates": g, "k3_classical_exact": {k: v["value"] for k, v in i["k3"].items()},
           "k2_classical_exact": {k: v["value"] for k, v in i["k2"].items()}, "milp_E6": milp, "k4_pilot": iii,
           "e3_k4_method": "exact (this code)" if iii["within_24h"] else "float32 search with double re-scoring, then brackets if > 24 h",
           "parts": {"i": i, "ii": ii}, "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("gates", "k3_classical_exact", "milp_E6", "e3_k4_method")}, indent=1))


if __name__ == "__main__":
    {"e2i": e2i, "e2ii": e2ii, "e2iis": e2iis, "e2iii": e2iii, "e2": e2}[sys.argv[1]]()
