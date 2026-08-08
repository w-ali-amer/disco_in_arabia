# -*- coding: utf-8 -*-
"""Exp 26 Phase 0: coherent disambiguation feasibility (design doc 15_).

Tests the load-bearing assumption of the one surviving application path:
conditioning a superposed joint reading-distribution on global constraints
via post-selection. THE number is p_post(k): if it collapses, the path dies.

Planted instances, k=1..8, chain vs loopy graphs, hard/soft/both layers.
Exactness gate: hard-only quantum conditional must equal classical posterior.
"""
import json
import numpy as np

RNG = np.random.default_rng(42)

def ry(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -s], [s, c]], dtype=complex)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def apply1(psi, U, q):
    psi = np.tensordot(U, psi, ([1], [q]))
    return np.moveaxis(psi, 0, q)

def apply2(psi, U4, qa, qb):
    T = U4.reshape(2, 2, 2, 2)
    psi = np.tensordot(T, psi, ([2, 3], [qa, qb]))
    return np.moveaxis(psi, [0, 1], [qa, qb])

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

def crz(t):
    return np.diag([1, 1, np.exp(-1j * np.pi * t),
                    np.exp(1j * np.pi * t)]).astype(complex)

def make_instance(k, loopy, seed):
    rng = np.random.default_rng(seed)
    r_star = rng.integers(0, 2, k)
    mislead = rng.choice(k, max(1, k // 2), replace=False)
    priors = np.where(np.isin(np.arange(k), mislead), 0.35, 0.65)
    # P(correct reading) per slot; convert to P(slot=1)
    p1 = np.where(r_star == 1, priors, 1 - priors)
    edges = [(i, i + 1) for i in range(k - 1)]
    if loopy and k >= 3:
        for i in range(0, k - 2, 2):
            edges.append((i, i + 2))
    parity = [(a, b, int(r_star[a] ^ r_star[b])) for a, b in edges]
    return r_star, p1, parity, mislead

def theta_for(p1):
    # Ry(t)|0> = cos(pi t)|0> + sin(pi t)|1>  ->  P(1) = sin^2(pi t)
    return np.arcsin(np.sqrt(p1)) / np.pi

def run_quantum(k, p1, parity, soft_phases=None, r_star=None):
    """Sequential ancilla conditioning. Returns (posterior over 2^k, p_post)."""
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    for i in range(k):
        psi = apply1(psi, ry(theta_for(p1[i])), i)
    p_post = 1.0
    for a, b, par in parity:
        # append ancilla |0>, CNOT from a and b, post-select on par
        psi = np.tensordot(psi, np.array([1, 0], dtype=complex), 0)
        anc = psi.ndim - 1
        psi = apply2(psi, CNOT, a, anc)
        psi = apply2(psi, CNOT, b, anc)
        psi = np.take(psi, par, axis=anc)
        n2 = float(np.sum(np.abs(psi) ** 2))
        if n2 < 1e-300:
            return None, 0.0
        psi = psi / np.sqrt(n2)
        p_post *= n2
    if soft_phases is not None:
        # sentence wire in |+>, reading-controlled CRz, H, post-select |0>
        psi = np.tensordot(psi, np.array([1, 1], dtype=complex) / np.sqrt(2), 0)
        sw = psi.ndim - 1
        for i in range(k):
            psi = apply2(psi, crz(soft_phases[i]), i, sw)
        psi = apply1(psi, H, sw)
        psi = np.take(psi, 0, axis=sw)
        n2 = float(np.sum(np.abs(psi) ** 2))
        if n2 < 1e-300:
            return None, 0.0
        psi = psi / np.sqrt(n2)
        p_post *= n2
    post = np.abs(psi.reshape(-1)) ** 2
    return post, p_post

def classical_posterior(k, p1, parity):
    post = np.zeros(2 ** k)
    for x in range(2 ** k):
        bits = [(x >> (k - 1 - i)) & 1 for i in range(k)]
        w = np.prod([p1[i] if bits[i] else 1 - p1[i] for i in range(k)])
        ok = all((bits[a] ^ bits[b]) == par for a, b, par in parity)
        post[x] = w * ok
    s = post.sum()
    return (post / s if s > 0 else post), s

def bits_to_idx(bits):
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x

def greedy_decode(p1):
    return (p1 > 0.5).astype(int)

def icm_decode(k, p1, parity, start):
    x = start.copy()
    for _ in range(50):
        changed = False
        for i in range(k):
            def score(v):
                pr = p1[i] if v else 1 - p1[i]
                pen = sum(1 for a, b, par in parity
                          if (a == i or b == i) and
                          ((x[a] if a != i else v) ^
                           (x[b] if b != i else v)) != par)
                return np.log(max(pr, 1e-9)) - 10.0 * pen
            best = int(score(1) > score(0))
            if best != x[i]:
                x[i] = best
                changed = True
        if not changed:
            break
    return x

OUT = {"pre_registered": "O1 exactness; O2 MAP beats greedy; O3 p_post decay; O4 soft layer"}
N_INST = 40
KS = list(range(1, 9))
results = {}
for loopy in (False, True):
    tag = "loopy" if loopy else "chain"
    res = {"k": KS, "p_post": [], "map_acc": [], "greedy_acc": [],
           "icm_acc": [], "exactness_max_err": [], "map_prob": [],
           "soft_map_acc": [], "soft_p_post": []}
    for k in KS:
        pp, mac, gac, iac, err, mpr, smac, spp = [], [], [], [], [], [], [], []
        for inst in range(N_INST):
            r_star, p1, parity, mislead = make_instance(k, loopy, 1000 * k + inst)
            q_post, p_post = run_quantum(k, p1, parity)
            c_post, c_mass = classical_posterior(k, p1, parity)
            if q_post is None:
                continue
            err.append(float(np.max(np.abs(q_post - c_post))))
            pp.append(p_post)
            map_idx = int(np.argmax(q_post))
            true_idx = bits_to_idx(r_star)
            mac.append(int(map_idx == true_idx))
            mpr.append(float(q_post[map_idx]))
            g = greedy_decode(p1)
            gac.append(int(bits_to_idx(g) == true_idx))
            iac.append(int(bits_to_idx(icm_decode(k, p1, parity, g)) == true_idx))
            # soft layer: phases +0.15 aligned with planted reading, else -0.15
            soft = [0.15 if r_star[i] == 1 else -0.15 for i in range(k)]
            sq_post, sp_post = run_quantum(k, p1, parity, soft_phases=soft)
            if sq_post is not None:
                smac.append(int(int(np.argmax(sq_post)) == true_idx))
                spp.append(sp_post)
        res["p_post"].append(float(np.mean(pp)))
        res["map_acc"].append(float(np.mean(mac)))
        res["greedy_acc"].append(float(np.mean(gac)))
        res["icm_acc"].append(float(np.mean(iac)))
        res["exactness_max_err"].append(float(np.max(err)))
        res["map_prob"].append(float(np.mean(mpr)))
        res["soft_map_acc"].append(float(np.mean(smac)))
        res["soft_p_post"].append(float(np.mean(spp)))
        print(f"[26] {tag} k={k}: p_post={np.mean(pp):.4f} "
              f"MAP={np.mean(mac):.2f} greedy={np.mean(gac):.2f} "
              f"ICM={np.mean(iac):.2f} exact_err={np.max(err):.1e} "
              f"softMAP={np.mean(smac):.2f} soft_pp={np.mean(spp):.4f}",
              flush=True)
    # decay fit: p_post ~ base^(k-1)
    lp = np.log(np.maximum(res["p_post"], 1e-300))
    base = float(np.exp(np.polyfit(np.array(KS) - 1, lp, 1)[0]))
    res["p_post_decay_base"] = base
    print(f"[26] {tag}: fitted p_post decay base per constraint ~ {base:.4f}",
          flush=True)
    results[tag] = res

# sizing table (O3): repetitions ~ samples_needed / p_post at k
for tag in results:
    base = results[tag]["p_post_decay_base"]
    tbl = {}
    for k in (10, 20, 50):
        n_constraints = (k - 1) + (0 if tag == "chain" else (k - 2 + 1) // 2)
        p_est = base ** n_constraints
        reps = 100 / max(p_est, 1e-300)  # ~100 post-selected samples for MAP
        tbl[k] = {"qubits_logical": k + 2, "p_post_est": p_est,
                  "repetitions_for_MAP": reps,
                  "hours_at_1e6_shots_per_hr": reps / 1e6}
    results[tag]["sizing"] = tbl
    print(f"[26] {tag} sizing: " + "; ".join(
        f"k={k}: {v['repetitions_for_MAP']:.2e} reps "
        f"({v['hours_at_1e6_shots_per_hr']:.2e} h @1M/hr)"
        for k, v in tbl.items()), flush=True)

OUT["results"] = results
json.dump(OUT, open("results_exp26.json", "w"), indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
for tag, mk in (("chain", "o-"), ("loopy", "s-")):
    a1.semilogy(KS, results[tag]["p_post"], mk, label=tag)
    a2.plot(KS, results[tag]["map_acc"], mk, label=f"{tag} conditioned MAP")
    a2.plot(KS, results[tag]["greedy_acc"], mk.replace("-", "--"),
            alpha=0.5, label=f"{tag} greedy")
a1.set_xlabel("k (ambiguous slots)")
a1.set_ylabel("post-selection success p_post")
a1.set_title("THE feasibility number: p_post(k)")
a1.legend()
a2.set_xlabel("k")
a2.set_ylabel("accuracy vs planted reading")
a2.set_title("Conditioned MAP vs greedy decoding")
a2.legend(fontsize=7)
fig.suptitle("Exp26 Phase 0 — coherent disambiguation feasibility")
fig.tight_layout()
fig.savefig("fig_exp26.png", dpi=200, bbox_inches="tight")
print("[26] DONE — results_exp26.json + fig_exp26.png", flush=True)
