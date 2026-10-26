# -*- coding: utf-8 -*-
"""exp46_descriptive.py -- doc 29 SS5 secondary analyses, DESCRIPTIVE ONLY.

These were specified in doc 29 SS5 and omitted from the first version of
exp46_stage_b.py. They are added here rather than folded into that script
because they must not touch the verdict: doc 29 SS5 registers them as
"secondary, descriptive only, never carrying a verdict", and the primary
statistic has already been computed. Recomputing nothing, this script reads
the committed distance matrix out of the results file.

  (i)  ROOT-OVERLAP CONFOUND CHECK. If root similarity rather than wazn
       tracks block geometry, that is an alternative explanation for any
       clustering. Mean distance is reported split by shared-radical count,
       plus a robustness recomputation of Delta and its exact permutation p
       with all >=2-radical-overlap pairs dropped. This matters most
       exactly when the primary p is marginal.

       CORRECTION TO doc 29 SS5, recorded here rather than silently: the doc
       named TWO high-overlap pairs (علّم/عالج and ذكّر/شكر). Enumerating the
       frozen inventory gives FIVE pairs sharing >=2 of 3 radicals, four of
       them ACROSS patterns:
         ذكّر(ذ.ك.ر)/شكر(ش.ك.ر)   cross-pattern
         راسل(ر.س.ل)/سأل(س.أ.ل)   cross-pattern
         ساعد(س.ع.د)/ودّع(و.د.ع)   cross-pattern
         عالج(ع.ل.ج)/علّم(ع.ل.م)   cross-pattern
         علّم(ع.ل.م)/كلّم(ك.ل.م)   same pattern (II)
       Direction of the bias matters and is not assumed: four of the five
       sit in the BETWEEN-pattern set, so if shared roots make blocks more
       similar they SHRINK the between-mean and bias Delta downward, i.e.
       against a positive. The numbers below decide it rather than the
       argument.
  (ii) CARTAN COORDINATES of the trained blocks -- the local-invariant
       fingerprint of a two-qubit gate. doc 26 SS3 carried this forward with
       the warning that Weyl-chamber canonicalisation is fiddly enough that
       it must not carry a criterion. Implemented via the standard
       magic-basis construction and SELF-TESTED against gates with known
       coordinates (identity, CNOT, SWAP) before any number is reported; if
       the self-test fails the section is reported as unavailable rather
       than as a result.

Nothing here can change wazn_clusters / wazn_clustering_ruled_out /
wazn_stage_b_inconclusive. Those stand as computed in the primary file.
"""

import argparse
import cmath
import glob
import itertools
import json
import math

import torch

import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note
from exp46_datagen import ROOTS, PATTERN_OF
from exp46_stage_b import all_partitions, delta

MAGIC = (1.0 / math.sqrt(2.0)) * torch.tensor(
    [[1, 0, 0, 1j],
     [0, 1j, 1, 0],
     [0, 1j, -1, 0],
     [1, 0, 0, -1j]], dtype=torch.complex128)


def shared_radicals(v, w):
    return len(set(ROOTS[v].split(".")) & set(ROOTS[w].split(".")))


# ------------------------------------------------------------ (ii) Cartan
def cartan_coords(U):
    """Canonical Cartan/KAK coordinates (c1,c2,c3) of a 2-qubit unitary.

    Standard construction: in the magic basis, M = Mg^dag U Mg has
    M^T M with eigenvalues exp(2i theta_k); the theta_k give the Cartan
    vector up to Weyl-chamber canonicalisation. DESCRIPTIVE ONLY."""
    U = U.to(torch.complex128)
    U = U / (torch.linalg.det(U) ** 0.25)          # to SU(4), phase-free
    M = MAGIC.conj().T @ U @ MAGIC
    A = M.T @ M
    ev = torch.linalg.eigvals(A)
    th = torch.tensor([cmath.phase(complex(e)) / 2.0 for e in ev],
                      dtype=torch.float64)
    th = torch.sort(th, descending=True).values
    # factor 1/2 fixed by the self-test below: without it CNOT came out at
    # pi/2 instead of the known pi/4 (and SWAP at pi/2 thrice instead of pi/4).
    c1 = float((th[0] + th[1]) / 2.0)
    c2 = float((th[0] + th[2]) / 2.0)
    c3 = float((th[1] + th[2]) / 2.0)
    v = sorted([abs(c1), abs(c2), abs(c3)], reverse=True)
    return [round(x, 6) for x in v]


def cartan_self_test():
    """Known coordinates, up to the canonicalisation used above:
    identity -> all ~0; CNOT -> one coordinate ~pi/4; SWAP -> three equal
    ~pi/4. Only the STRUCTURE is asserted, not a convention-specific sign."""
    eye = torch.eye(4, dtype=torch.complex128)
    cnot = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=torch.complex128)
    swap = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=torch.complex128)
    ci, cc, cs = (cartan_coords(eye), cartan_coords(cnot),
                  cartan_coords(swap))
    q = math.pi / 4
    ok_i = max(ci) < 1e-6
    ok_c = abs(cc[0] - q) < 1e-5 and cc[1] < 1e-5
    ok_s = all(abs(x - q) < 1e-5 for x in cs)
    return {"identity": ci, "cnot": cc, "swap": cs,
            "identity_ok": bool(ok_i), "cnot_ok": bool(ok_c),
            "swap_ok": bool(ok_s),
            "self_test_passed": bool(ok_i and ok_c and ok_s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--arm", default="A3")
    args = ap.parse_args()

    path = args.results
    if path is None:
        cands = sorted(glob.glob("results_exp46_stage_b_%s*.json" % args.arm))
        if not cands:
            print("[exp46-desc] no primary results file yet; nothing to do.")
            return 1
        path = cands[-1]
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    if res.get("verdicts", {}).get("exp46_void"):
        print("[exp46-desc] primary run is VOID; descriptives not reported.")
        return 1

    verbs = res["verbs_sorted"]
    D = res["distance_matrix_seed_averaged"]
    n = len(verbs)
    idx = {v: i for i, v in enumerate(verbs)}

    # -------- (i) root overlap
    buckets = {0: [], 1: [], 2: [], 3: []}
    cross_pattern_high = []
    for a, b in itertools.combinations(verbs, 2):
        k = shared_radicals(a, b)
        d = D[idx[a]][idx[b]]
        buckets[k].append(d)
        same = PATTERN_OF[a] == PATTERN_OF[b]
        if k >= 2:
            cross_pattern_high.append(
                {"pair": [a, b], "roots": [ROOTS[a], ROOTS[b]],
                 "shared_radicals": k, "same_pattern": bool(same),
                 "distance": d})
    by_overlap = {str(k): {"n_pairs": len(v),
                           "mean_distance": (sum(v) / len(v)) if v else None}
                  for k, v in buckets.items()}

    # robustness: drop every pair with >=2 shared radicals, recompute
    drop = {(idx[p["pair"][0]], idx[p["pair"][1]]) for p in cross_pattern_high}
    Dm = [[D[i][j] for j in range(n)] for i in range(n)]
    kept_mask = [[True] * n for _ in range(n)]
    for i, j in drop:
        kept_mask[i][j] = kept_mask[j][i] = False

    def delta_masked(mat, part):
        grp = {}
        for gi, g in enumerate(part):
            for e in g:
                grp[e] = gi
        w, b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if not kept_mask[i][j]:
                    continue
                (w if grp[i] == grp[j] else b).append(mat[i][j])
        if not w or not b:
            return None
        return sum(b) / len(b) - sum(w) / len(w)

    parts = all_partitions()
    observed = tuple(tuple(res["observed_partition_indices"][k])
                     for k in range(3))
    obs_m = delta_masked(Dm, observed)
    nulls = [x for x in (delta_masked(Dm, p) for p in parts) if x is not None]
    p_m = sum(1 for v in nulls if v >= obs_m) / len(nulls)

    # -------- (ii) Cartan
    st = cartan_self_test()
    cartan = {"self_test": st}
    if st["self_test_passed"]:
        per_verb = {}
        ck = sorted(glob.glob("exp46_ckpt/w1_%s_s*_lr0.02_b8.json" % args.arm))
        used = 0
        for j in ck:
            pt = j[:-5] + ".pt"
            try:
                sd = torch.load(pt, weights_only=True)
            except Exception:
                continue
            m = QuantumStoryModel(args.arm, verbs, 0)
            m.load_state_dict(sd)
            with torch.no_grad():
                for v in verbs:
                    per_verb.setdefault(v, []).append(
                        cartan_coords(m.verb_matrix(v)))
            used += 1
        cartan["n_checkpoints"] = used
        cartan["per_verb_mean"] = {
            v: [sum(c[k] for c in cs) / len(cs) for k in range(3)]
            for v, cs in per_verb.items()} if used else None
        cartan["note"] = ("Weyl-chamber canonicalisation is fiddly (doc 26 "
                          "SS3); reported for interpretation only and "
                          "carrying no criterion.")
    else:
        cartan["note"] = ("SELF-TEST FAILED -- Cartan section reported as "
                          "UNAVAILABLE rather than as a result.")

    out = {
        "experiment": "exp46 Stage B -- doc 29 SS5 secondary descriptives",
        "status": ("DESCRIPTIVE ONLY. Cannot and does not change "
                   "wazn_clusters / wazn_clustering_ruled_out / "
                   "wazn_stage_b_inconclusive, which stand as computed in "
                   "the primary results file."),
        "primary_results_file": path,
        "primary_verdicts": res.get("verdicts"),
        "primary_statistic": res.get("primary_statistic"),
        "root_overlap": {
            "mean_distance_by_shared_radical_count": by_overlap,
            "pairs_with_2plus_shared_radicals": cross_pattern_high,
            "robustness_excluding_those_pairs": {
                "delta": obs_m, "p": p_m, "n_pairs_dropped": len(drop),
                "note": ("recomputes the primary statistic with every "
                         ">=2-shared-radical pair removed. DESCRIPTIVE: it "
                         "shows whether root similarity rather than wazn "
                         "carries the signal. It does not replace the "
                         "registered criterion.")},
        },
        "cartan_coordinates": cartan,
        "git_note": git_note(),
    }
    p = fresh_results_path("results_exp46_descriptive_%s" % args.arm)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("[exp46-desc] mean distance by shared-radical count:")
    for k in sorted(by_overlap):
        b = by_overlap[k]
        if b["n_pairs"]:
            print("   %s radicals: n=%2d mean=%+.5f"
                  % (k, b["n_pairs"], b["mean_distance"]))
    print("[exp46-desc] cross-pattern high-overlap pairs:")
    for c in cross_pattern_high:
        print("   %s/%s (%s,%s) shared=%d same_pattern=%s d=%+.5f"
              % (c["pair"][0], c["pair"][1], c["roots"][0], c["roots"][1],
                 c["shared_radicals"], c["same_pattern"], c["distance"]))
    print("[exp46-desc] robustness (drop >=2-overlap pairs): delta=%+.5f p=%.4f"
          % (obs_m, p_m))
    print("[exp46-desc] Cartan self-test passed: %s" % st["self_test_passed"])
    print("[exp46-desc] wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
