"""Exp 15c: CORRECTED Rung-2. True multiset twins as condition A; two weight schemes:
W1 = original (hash-fallback, as exp13/exp15 ran) -> isolates the pairing fix
W2 = repaired AraVec lookup (strip morph tags) -> value-tied, content-bearing params, the real H1 test
"""
import json, time, itertools
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
import math, hashlib

RNG = np.random.default_rng(42)
data = json.load(open("sentences.json"))["WordOrderMatched"]
sents  = [d["sentence"] for d in data]
labels = [d["label"] for d in data]

# --- true twin pairing by word multiset ---
svo_i = [i for i, l in enumerate(labels) if l.endswith("SVO")]
vso_i = [i for i, l in enumerate(labels) if l.endswith("VSO")]
vpool = defaultdict(list)
for i in vso_i: vpool[tuple(sorted(sents[i].split()))].append(i)
twins = []
for i in svo_i:
    k = tuple(sorted(sents[i].split()))
    if vpool[k]: twins.append((i, vpool[k].pop(0)))
print(f"[15c] true twins: {len(twins)}/60")

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)
ansatz = exp13.make_ansatz(1, 1)
circs, vidx = [], []
for i, d in enumerate(diagrams):
    if d is None: continue
    try:
        circs.append(ansatz(exp13._remove_cups(d))); vidx.append(i)
    except Exception: pass
pos = {orig: k for k, orig in enumerate(vidx)}
print(f"[15c] circuits: {len(circs)}")

model = NumpyModel.from_diagrams(circs, use_jit=False)
names = [str(s) for s in model.symbols]

def weights_W1():
    return exp13.warmstart_weights(model)

def weights_W2():
    w = np.empty(len(names)); n_hit = 0
    for i, name in enumerate(names):
        word_part = name.split("__")[0]
        base = word_part.split("_")[0]
        try: idx = int(name.rsplit("_", 1)[-1])
        except ValueError: idx = 0
        vec = exp13._aravec_vec(base)
        if vec is not None:
            w[i] = (float(vec[idx % len(vec)]) + 1.0) * math.pi; n_hit += 1
        else:
            h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            w[i] = (h / 0xFFFFFFFF) * 2 * math.pi
    print(f"[15c] W2 aravec hits: {n_hit}/{len(names)} ({100*n_hit/len(names):.1f}%)")
    return w

def states_for(weights):
    wmap = dict(zip(names, weights))
    S, P = [], []
    for c in circs:
        syms = sorted(c.free_symbols, key=str)
        vals = [wmap[str(s)] for s in syms]
        amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
        p = float(np.sum(np.abs(amps) ** 2)); P.append(p)
        S.append(amps / np.sqrt(p) if p > 1e-12 else amps * np.nan)
    return np.array(S), np.array(P)

def fid(a, b): return float(np.abs(np.vdot(a, b)) ** 2)

# conditions in circuit-index space
A = [(pos[i], pos[j]) for i, j in twins if i in pos and j in pos]
svo_k = [pos[i] for i in svo_i if i in pos]; vso_k = [pos[i] for i in vso_i if i in pos]
twinset = set((min(a,b), max(a,b)) for a, b in A)
def sample_same(pool, n):
    out, seen = [], set()
    while len(out) < n:
        i, j = RNG.choice(pool, 2, replace=False)
        key = (min(i,j), max(i,j))
        if key in seen: continue
        seen.add(key); out.append((int(i), int(j)))
    return out
B = sample_same(svo_k, 30) + sample_same(vso_k, 30)
C, seen = [], set()
while len(C) < 60:
    i = int(RNG.choice(svo_k)); j = int(RNG.choice(vso_k))
    key = (min(i,j), max(i,j))
    if key in seen or key in twinset: continue
    seen.add(key); C.append((i, j))

from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

def analyse(S, tag):
    fA = np.array([fid(S[i], S[j]) for i, j in A])
    fB = np.array([fid(S[i], S[j]) for i, j in B])
    fC = np.array([fid(S[i], S[j]) for i, j in C])
    out = dict(medians=dict(A=float(np.median(fA)), B=float(np.median(fB)), C=float(np.median(fC))), nA=len(fA))
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        y = np.r_[np.ones(len(fA)), np.zeros(len(neg))]; s = np.r_[fA, neg]
        obs = roc_auc_score(y, s)
        perm = np.array([roc_auc_score(RNG.permutation(y), s) for _ in range(10000)])
        pval = float((np.sum(perm >= obs) + 1) / 10001)
        U, pu = mannwhitneyu(fA, neg, alternative="greater")
        out[nm] = dict(auc=float(obs), perm_p=pval, mwu_p=float(pu))
    # retrieval vs true twin
    partner = {}
    for i, j in A: partner[i] = j; partner[j] = i
    n = len(S); F = np.array([[fid(S[i], S[j]) for j in range(n)] for i in range(n)])
    top1, rr = 0, []
    for i in range(n):
        if i not in partner: continue
        others = np.delete(np.arange(n), i)
        order = others[np.argsort(-np.delete(F[i], i))]
        rank = int(np.where(order == partner[i])[0][0]) + 1
        rr.append(1.0 / rank); top1 += int(rank == 1)
    out["retrieval"] = dict(top1=top1 / len(rr), mrr=float(np.mean(rr)), n=len(rr))
    print(f"[15c] {tag}: medians A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} C={out['medians']['C']:.4f} | AUC A-vs-B={out['A_vs_B']['auc']:.4f} (p={out['A_vs_B']['perm_p']:.5f}) | top1={out['retrieval']['top1']:.3f} MRR={out['retrieval']['mrr']:.3f}")
    return out

OUT = {"n_true_twins": len(A)}
S1, P1 = states_for(weights_W1())
OUT["W1_original_hash_weights"] = analyse(S1, "W1 (original hash weights, corrected pairs)")
S2, P2 = states_for(weights_W2())
OUT["W2_aravec_tied_weights"] = analyse(S2, "W2 (repaired AraVec tied weights, corrected pairs)")
np.savez("states_L1_tied.npz", states=S2, norms=P2, vidx=vidx,
         labels=np.array([labels[i] for i in vidx]),
         sentences=np.array([sents[i] for i in vidx]),
         twin_pairs=np.array(A))
OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp15c.json", "w"), indent=2, ensure_ascii=False)
print(f"[15c] DONE in {OUT['runtime_sec']}s")
