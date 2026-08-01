"""Exp 17: CAPACITY test — does meaning proximity appear when the sentence
space widens? Rung-2 follow-up (the pre-registered row-4 response in spec 04).

Same protocol as exp15c (true multiset twins; conditions A twin / B same-order
non-twin / C cross-order; AUC + 10k-permutation test + retrieval), but with
n_s_qubits in {2, 3} — 4- and 8-dimensional sentence states instead of 2.
Weight schemes via the repaired exp13 warmstart: W1 = legacy hash,
W2 = fixed AraVec-tied. L0 exactness control re-checked at each width.

Pre-registered predictions:
- H-cap: if the exp15c null was a capacity bottleneck, W2 AUC(A vs B) should
  rise above 0.5 at n_s >= 2. If it stays ~0.5 at both widths, the bottleneck
  is type-disjoint parameter routing -> functorial tying (Rung 3).
- L0 control: all states identical at every width (architectural theorem).
"""
import json, time, os
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

RNG = np.random.default_rng(42)
data = json.load(open("sentences.json"))["WordOrderMatched"]
sents = [d["sentence"] for d in data]
labels = [d["label"] for d in data]

svo_i = [i for i, l in enumerate(labels) if l.endswith("SVO")]
vso_i = [i for i, l in enumerate(labels) if l.endswith("VSO")]
vpool = defaultdict(list)
for i in vso_i:
    vpool[tuple(sorted(sents[i].split()))].append(i)
twins = []
for i in svo_i:
    k = tuple(sorted(sents[i].split()))
    if vpool[k]:
        twins.append((i, vpool[k].pop(0)))
print(f"[17] true twins: {len(twins)}/60", flush=True)

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)


def circuits_for(n_layers, n_s):
    ansatz = exp13.make_ansatz(n_layers, n_s)
    circs, vidx = [], []
    for i, d in enumerate(diagrams):
        if d is None:
            continue
        try:
            circs.append(ansatz(exp13._remove_cups(d)))
            vidx.append(i)
        except Exception:
            pass
    return circs, vidx


def states_for(circs, weights, names):
    wmap = dict(zip(names, weights))
    S, P = [], []
    for c in circs:
        syms = sorted(c.free_symbols, key=str)
        vals = [wmap[str(s)] for s in syms]
        amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
        p = float(np.sum(np.abs(amps) ** 2))
        P.append(p)
        S.append(amps / np.sqrt(p) if p > 1e-12 else amps * np.nan)
    return np.array(S), np.array(P)


def fid(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)


def sample_same(pool, n):
    out, seen = [], set()
    while len(out) < n:
        i, j = RNG.choice(pool, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(i), int(j)))
    return out


def analyse(S, A, B, C, tag):
    fA = np.array([fid(S[i], S[j]) for i, j in A])
    fB = np.array([fid(S[i], S[j]) for i, j in B])
    fC = np.array([fid(S[i], S[j]) for i, j in C])
    out = dict(medians=dict(A=float(np.median(fA)), B=float(np.median(fB)),
                            C=float(np.median(fC))), nA=len(fA))
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        y = np.r_[np.ones(len(fA)), np.zeros(len(neg))]
        s = np.r_[fA, neg]
        obs = roc_auc_score(y, s)
        perm = np.array([roc_auc_score(RNG.permutation(y), s) for _ in range(10000)])
        pval = float((np.sum(perm >= obs) + 1) / 10001)
        U, pu = mannwhitneyu(fA, neg, alternative="greater")
        out[nm] = dict(auc=float(obs), perm_p=pval, mwu_p=float(pu))
    partner = {}
    for i, j in A:
        partner[i] = j
        partner[j] = i
    n = len(S)
    F = np.array([[fid(S[i], S[j]) for j in range(n)] for i in range(n)])
    top1, rr = 0, []
    for i in range(n):
        if i not in partner:
            continue
        others = np.delete(np.arange(n), i)
        order = others[np.argsort(-np.delete(F[i], i))]
        rank = int(np.where(order == partner[i])[0][0]) + 1
        rr.append(1.0 / rank)
        top1 += int(rank == 1)
    out["retrieval"] = dict(top1=top1 / len(rr), mrr=float(np.mean(rr)), n=len(rr))
    print(f"[17] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC A-vs-B={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) | top1={out['retrieval']['top1']:.3f} "
          f"MRR={out['retrieval']['mrr']:.3f}", flush=True)
    return out


OUT = {"n_true_twins": len(twins)}
for n_s in (2, 3):
    c1, v1 = circuits_for(1, n_s)
    print(f"[17] n_s={n_s}: {len(c1)} circuits", flush=True)
    pos = {orig: k for k, orig in enumerate(v1)}
    A = [(pos[i], pos[j]) for i, j in twins if i in pos and j in pos]
    svo_k = [pos[i] for i in svo_i if i in pos]
    vso_k = [pos[i] for i in vso_i if i in pos]
    twinset = set((min(a, b), max(a, b)) for a, b in A)
    B = sample_same(svo_k, 30) + sample_same(vso_k, 30)
    C, seen = [], set()
    while len(C) < 60:
        i = int(RNG.choice(svo_k))
        j = int(RNG.choice(vso_k))
        key = (min(i, j), max(i, j))
        if key in seen or key in twinset:
            continue
        seen.add(key)
        C.append((i, j))

    model = NumpyModel.from_diagrams(c1, use_jit=False)
    names = [str(s) for s in model.symbols]
    res_ns = {}
    for mode, key in (("legacy", "W1_hash"), ("fixed", "W2_aravec_tied")):
        os.environ["QFM_WARMSTART"] = mode
        w = exp13.warmstart_weights(model)
        S, P = states_for(c1, w, names)
        res_ns[key] = analyse(S, A, B, C, f"n_s={n_s} {key}")
        if mode == "fixed":
            np.savez(f"states_L1_ns{n_s}.npz", states=S, norms=P, vidx=v1,
                     labels=np.array([labels[i] for i in v1]),
                     sentences=np.array([sents[i] for i in v1]),
                     twin_pairs=np.array(A))

    c0, v0 = circuits_for(0, n_s)
    m0 = NumpyModel.from_diagrams(c0, use_jit=False)
    w0 = exp13.warmstart_weights(m0)
    S0, _ = states_for(c0, w0, [str(s) for s in m0.symbols])
    f0 = [fid(S0[0], S0[k]) for k in range(1, len(S0))]
    res_ns["L0_control_min_fid_vs_first"] = float(np.min(f0))
    print(f"[17] n_s={n_s} L0 control min fidelity = {np.min(f0):.12f}", flush=True)
    OUT[f"n_s_{n_s}"] = res_ns
    json.dump(OUT, open("results_exp17.json", "w"), indent=2)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp17.json", "w"), indent=2)
print(f"[17] DONE in {OUT['runtime_sec']}s", flush=True)
