"""Exp 18: Rung-3 Stage 1 — typed-PCA embedding->parameter encoder.

The missing methodology: every existing QNLP warmstart (incl. lambeq's and
our exp13 'fixed' mode) index-samples embedding components, preserving none
of the embedding geometry. Stage-1 encoder: for each grammatical type, fit a
PCA over the AraVec vectors of all vocabulary words carrying that type, with
as many components as the type has circuit parameters; a word's parameters
are its principal-component scores, each component min-max scaled to
[0, 2pi). Deterministic, geometry-preserving, drop-in.

Evaluation: exp17 geometry protocol (true twins; A twin / B same-order /
C cross-order; AUC + 10k permutations + retrieval) at n_s_qubits = 2 (the
width where content anchoring first became visible: index-sampled W2 AUC
0.569 p=0.095) and n_s=1 (exp15c baseline: 0.499). Pre-registered question:
does a geometry-preserving encoder beat index-sampling on twin proximity?
AraVec lookups use the normalized dump (150/150 coverage).
"""
import json, time, os, hashlib, math
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

RNG = np.random.default_rng(42)
data = json.load(open("sentences.json"))["WordOrderMatched"]
sents = [d["sentence"] for d in data]
labels = [d["label"] for d in data]
WVEC = json.load(open("exp16_wordvecs.json"))["vectors"]  # 150/150 normalized

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
print(f"[18] twins: {len(twins)}", flush=True)

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)


def circuits_for(n_s):
    ansatz = exp13.make_ansatz(1, n_s)
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


def parse_symbol(name):
    """'الولد_NUM-s_GEN-m__n_0' -> (base='الولد', type='n', idx=0)."""
    word_part, rest = name.split("__", 1)
    base = word_part.split("_")[0]
    type_str, idx_s = rest.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        type_str, idx = rest, 0
    return base, type_str, idx


def typed_pca_weights(names):
    """Stage-1 encoder. Returns weights + coverage stats."""
    per_type = defaultdict(set)          # type -> bases
    need = defaultdict(int)              # type -> max param index + 1
    parsed = [parse_symbol(nm) for nm in names]
    for base, tstr, idx in parsed:
        per_type[tstr].add(base)
        need[tstr] = max(need[tstr], idx + 1)
    scores = {}                          # (type) -> {base: scaled comp vector}
    info = {}
    for tstr, bases in per_type.items():
        bases = sorted(bases)
        vecs = np.array([WVEC[b] for b in bases if b in WVEC])
        have = [b for b in bases if b in WVEC]
        k_req = need[tstr]
        k = int(min(k_req, len(have) - 1 if len(have) > 1 else 1, vecs.shape[1])) \
            if len(have) else 0
        comp = {}
        if k >= 1:
            pca = PCA(n_components=k, random_state=0)
            S = pca.fit_transform(vecs)          # (n_words, k)
            lo, hi = S.min(axis=0), S.max(axis=0)
            span = np.where(hi > lo, hi - lo, 1.0)
            S01 = (S - lo) / span                # each component -> [0,1]
            for b, row in zip(have, S01):
                comp[b] = row * 2 * np.pi
        scores[tstr] = comp
        info[tstr] = {"n_words": len(bases), "n_with_vec": len(have),
                      "params_needed": k_req, "pca_components": k,
                      "evr_sum": float(np.sum(
                          pca.explained_variance_ratio_)) if k >= 1 else 0.0}
    w = np.empty(len(names))
    n_pca = n_fb = 0
    for i, (nm, (base, tstr, idx)) in enumerate(zip(names, parsed)):
        row = scores.get(tstr, {}).get(base)
        if row is not None and idx < len(row):
            w[i] = float(row[idx]); n_pca += 1
        else:
            h = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
            w[i] = (h / 0xFFFFFFFF) * 2 * math.pi; n_fb += 1
    print(f"[18] encoder coverage: {n_pca}/{len(names)} PCA-assigned, "
          f"{n_fb} hash-fallback", flush=True)
    return w, {"assigned": n_pca, "fallback": n_fb, "types": info}


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
        seen.add(key); out.append((int(i), int(j)))
    return out


def analyse(S, A, B, C, tag):
    fA = np.array([fid(S[i], S[j]) for i, j in A])
    fB = np.array([fid(S[i], S[j]) for i, j in B])
    fC = np.array([fid(S[i], S[j]) for i, j in C])
    out = dict(medians=dict(A=float(np.median(fA)), B=float(np.median(fB)),
                            C=float(np.median(fC))))
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        y = np.r_[np.ones(len(fA)), np.zeros(len(neg))]
        s = np.r_[fA, neg]
        obs = roc_auc_score(y, s)
        perm = np.array([roc_auc_score(RNG.permutation(y), s)
                         for _ in range(10000)])
        pval = float((np.sum(perm >= obs) + 1) / 10001)
        U, pu = mannwhitneyu(fA, neg, alternative="greater")
        out[nm] = dict(auc=float(obs), perm_p=pval, mwu_p=float(pu))
    partner = {}
    for i, j in A:
        partner[i] = j; partner[j] = i
    nS = len(S)
    F = np.array([[fid(S[i], S[j]) for j in range(nS)] for i in range(nS)])
    top1, rr = 0, []
    for i in range(nS):
        if i not in partner:
            continue
        others = np.delete(np.arange(nS), i)
        order = others[np.argsort(-np.delete(F[i], i))]
        rank = int(np.where(order == partner[i])[0][0]) + 1
        rr.append(1.0 / rank); top1 += int(rank == 1)
    out["retrieval"] = dict(top1=top1 / len(rr), mrr=float(np.mean(rr)))
    print(f"[18] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC A-vs-B={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) | top1={out['retrieval']['top1']:.3f} "
          f"MRR={out['retrieval']['mrr']:.3f}", flush=True)
    return out


OUT = {"n_true_twins": len(twins),
       "baselines": {"ns1_index_sampled_W2": {"auc": 0.499, "p": 0.51},
                     "ns2_index_sampled_W2": {"auc": 0.569, "p": 0.095},
                     "ns2_hash_W1": {"auc": 0.649, "p": 0.002}}}
for n_s in (1, 2):
    circs, vidx = circuits_for(n_s)
    pos = {orig: k for k, orig in enumerate(vidx)}
    A = [(pos[i], pos[j]) for i, j in twins if i in pos and j in pos]
    svo_k = [pos[i] for i in svo_i if i in pos]
    vso_k = [pos[i] for i in vso_i if i in pos]
    twinset = set((min(a, b), max(a, b)) for a, b in A)
    B = sample_same(svo_k, 30) + sample_same(vso_k, 30)
    C, seen = [], set()
    while len(C) < 60:
        i = int(RNG.choice(svo_k)); j = int(RNG.choice(vso_k))
        key = (min(i, j), max(i, j))
        if key in seen or key in twinset:
            continue
        seen.add(key); C.append((i, j))
    model = NumpyModel.from_diagrams(circs, use_jit=False)
    names = [str(s) for s in model.symbols]
    w, cov = typed_pca_weights(names)
    S, P = states_for(circs, w, names)
    OUT[f"n_s_{n_s}"] = {"encoder_coverage": cov,
                         "geometry": analyse(S, A, B, C,
                                             f"typed-PCA n_s={n_s}")}
    if n_s == 2:
        np.savez("states_L1_ns2_typedpca.npz", states=S, norms=P, vidx=vidx,
                 labels=np.array([labels[i] for i in vidx]),
                 sentences=np.array([sents[i] for i in vidx]),
                 twin_pairs=np.array(A))
    json.dump(OUT, open("results_exp18.json", "w"), indent=2)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp18.json", "w"), indent=2)
print(f"[18] DONE in {OUT['runtime_sec']}s", flush=True)
