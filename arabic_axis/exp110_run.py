# -*- coding: utf-8 -*-
"""exp110_run.py -- doc 110 (REGISTERED 5886cfc): which-referent games for adjective attachment.

U: uniform, C = 6, K in {2,3,4}, ordered distinct K-tuples, question uniform over the tuple's classes, answer = position;
   m in {1,2}: classical exact (MILP) vs quantum (exp100 general optimiser, D = 2^m); classical perfect-service bits
   (smallest m <= 3 with classical exact = 1).
E: empirical rows from chains_exp110.json split (a) (primary) and (a)+(b) (descriptive); m in {1,2}.
Gate attachment_qrac_empirical: quantum - classical >= 0.05 on E(a) at m = 1 or 2.  Writes results_exp110.json (Rule 3).
"""
import collections, itertools, json
import exp94_common as C94
import exp100_accounting as A100

CLASSES = ["ms", "fs", "md", "fd", "mp", "fp"]


def uniform_rows(K):
    rows = []; tuples = list(itertools.permutations(CLASSES, K)); w = 1.0 / (len(tuples) * K)
    for t in tuples:
        for pos, v in enumerate(t):
            rows.append((t, v, pos, w))
    return rows


def run(rows, ms=(1, 2), perfect=False):
    pb = A100.problem(rows); out = {"n_x": len(pb["X"])}
    for m in ms:
        cl = A100.classical_milp(pb, m, 3600); q = A100.quantum(pb, m)
        cval = cl["value"] if cl["proved_optimal"] else cl["dual_bound"]
        out["m%d" % m] = {"classical": cl["value"], "classical_proved": cl["proved_optimal"], "classical_dual": cl["dual_bound"],
                          "quantum": q["value"], "G_POVM": q["G_POVM"], "margin": q["value"] - cval}
        print(" m", m, out["m%d" % m], flush=True)
    if perfect:
        for m in (1, 2, 3):
            cl = A100.classical_milp(pb, m, 3600)
            if cl["proved_optimal"] and cl["value"] >= 1 - 1e-9:
                out["perfect_service_bits"] = m; break
        else:
            out["perfect_service_bits"] = ">3"
    return out


def main():
    ch = json.load(open("chains_exp110.json")); out = {"doc": "110 (5886cfc)", "U": {}, "E": {}}
    for K in (2, 3, 4):
        print("U K", K, flush=True); out["U"]["K%d" % K] = run(uniform_rows(K), perfect=True)
    rows_a = [(tuple(r["x"]), r["r"], r["gold"], 1.0) for r in ch["rows"] if r["kind"] == "a"]
    rows_ab = [(tuple(r["x"]), r["r"], r["gold"], 1.0) for r in ch["rows"] if r["kind"] in ("a", "b")]
    print("E(a)", len(rows_a), flush=True); out["E"]["a_primary"] = run(rows_a)
    print("E(a+b)", len(rows_ab), flush=True); out["E"]["ab_descriptive"] = run(rows_ab)
    b = [r for r in ch["rows"] if r["kind"] == "b"]
    out["tied_stats"] = {"n": len(b), "proximity_rate": sum(r["proximity_correct"] for r in b) / max(1, len(b)),
                         "case_breaks_tie_rate": sum(r["case_breaks_tie"] for r in b) / max(1, len(b))}
    out["chain_stats"] = ch["stats"]; out["K_dist"] = ch["K_dist"]
    m1, m2 = out["E"]["a_primary"]["m1"]["margin"], out["E"]["a_primary"]["m2"]["margin"]
    out["attachment_qrac_empirical"] = bool(max(m1, m2) >= 0.05)
    out["G_POVM"] = all(v["G_POVM"] for d in list(out["U"].values()) + list(out["E"].values()) for k, v in d.items() if k.startswith("m"))
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp110.json", out)
    print(json.dumps({"gate": out["attachment_qrac_empirical"], "tied": out["tied_stats"], "G_POVM": out["G_POVM"]}), flush=True)


if __name__ == "__main__":
    main()
