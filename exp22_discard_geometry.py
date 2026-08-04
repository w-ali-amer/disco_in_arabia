"""Exp 22: discard geometry — was the semantic-geometry null an artifact of
post-selection? Same twin protocol, but sentences become DENSITY MATRICES
(discard=True: partial trace instead of Bra(0)), compared by Uhlmann fidelity.

Pre-registered: L0-discard control must give all-identical DMs; if
post-selection was the flaw, twin AUC > 0.5 (p<0.05) under content-bearing
weights; a persistent null extends the negative to mixed states.
Design doc: 14_exp22_24_design.md.
"""
import json, os, time
import numpy as np
from collections import defaultdict
from scipy.linalg import sqrtm

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
print(f"[22] twins: {len(twins)}", flush=True)

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)


def circuits_for(n_layers, n_s):
    ansatz = exp13.IQPAnsatz({exp13.S_ty: n_s, exp13.N_ty: 1},
                             n_layers=n_layers, discard=True)
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


def to_dm(T, d, is_mixed):
    """Tensor -> d x d density matrix; auto-detect index ordering.
    Pure outputs (circuits that needed no Discard boxes) are lifted to
    rank-1 density matrices."""
    T = np.asarray(T)
    if not is_mixed:
        v = T.flatten()
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-12 else v
        return np.outer(v, v.conj()), n ** 2
    cands = [T.reshape(d, d)]
    if T.ndim == 4:  # (2,2,2,2): try [k1,b1,k2,b2] interleaved ordering too
        cands.append(np.transpose(T, (0, 2, 1, 3)).reshape(d, d))
    for rho in cands:
        herm = np.linalg.norm(rho - rho.conj().T) < 1e-8
        ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
        if herm and ev.min() > -1e-8:
            tr = float(np.real(np.trace(rho)))
            return rho / tr if tr > 1e-12 else rho, tr
    raise ValueError("no Hermitian PSD ordering found")


def dms_for(circs, wmap, dim):
    out, traces, n_mixed = [], [], 0
    for c in circs:
        syms = sorted(c.free_symbols, key=str)
        vals = [wmap[str(s)] for s in syms]
        T = (c.lambdify(*syms)(*vals).eval() if syms else c.eval())
        rho, tr = to_dm(T, dim, c.is_mixed)
        n_mixed += int(c.is_mixed)
        out.append(rho)
        traces.append(tr)
    print(f"[22]   ({n_mixed}/{len(circs)} circuits mixed, rest pure)",
          flush=True)
    return out, traces


def uhlmann_cache(rhos):
    return [sqrtm(r) for r in rhos]


def uhl(sq_i, rho_j):
    m = sqrtm(sq_i @ rho_j @ sq_i)
    f = float(np.real(np.trace(m))) ** 2
    return min(max(f, 0.0), 1.0 + 1e-9)


def analyse(rhos, A, B, C, tag):
    sq = uhlmann_cache(rhos)
    fA = np.array([uhl(sq[i], rhos[j]) for i, j in A])
    fB = np.array([uhl(sq[i], rhos[j]) for i, j in B])
    fC = np.array([uhl(sq[i], rhos[j]) for i, j in C])
    out = {"medians": {"A": float(np.median(fA)), "B": float(np.median(fB)),
                       "C": float(np.median(fC))}}
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        y = np.r_[np.ones(len(fA)), np.zeros(len(neg))]
        s = np.r_[fA, neg]
        obs = roc_auc_score(y, s)
        prm = np.array([roc_auc_score(RNG.permutation(y), s)
                        for _ in range(10000)])
        out[nm] = {"auc": float(obs),
                   "perm_p": float((np.sum(prm >= obs) + 1) / 10001),
                   "mwu_p": float(mannwhitneyu(fA, neg,
                                               alternative="greater")[1])}
    print(f"[22] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC_AB={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) AUC_AC={out['A_vs_C']['auc']:.4f} "
          f"(p={out['A_vs_C']['perm_p']:.5f})", flush=True)
    return out


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


try:
    npz = np.load("exp21a_weights.npz", allow_pickle=True)
    AE_MAP = dict(zip(list(npz["names"]), npz["w0"].astype(float)))
except Exception:
    AE_MAP = {}

OUT = {"n_true_twins": len(twins)}
for n_s in (1, 2):
    circs, vidx = circuits_for(1, n_s)
    print(f"[22] n_s={n_s}: {len(circs)} discard circuits", flush=True)
    pos = {orig: k for k, orig in enumerate(vidx)}
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
    model = NumpyModel.from_diagrams(circs, use_jit=False)
    names = [str(s) for s in model.symbols]
    res = {}
    schemes = [("hash", "legacy"), ("fixed", "fixed")]
    for tag, mode in schemes:
        os.environ["QFM_WARMSTART"] = mode
        w = exp13.warmstart_weights(model)
        wmap = dict(zip(names, w))
        rhos, traces = dms_for(circs, wmap, 2 ** n_s)
        res[tag] = analyse(rhos, A, B, C, f"n_s={n_s} {tag}")
        res[tag]["mean_trace"] = float(np.mean(traces))
    if n_s == 2 and AE_MAP:
        import hashlib, math
        w = np.array([AE_MAP.get(nm,
                                 (int(hashlib.md5(nm.encode()).hexdigest()[:8],
                                      16) / 0xFFFFFFFF) * 2 * math.pi)
                      for nm in names])
        n_hit = sum(nm in AE_MAP for nm in names)
        print(f"[22] ae_init name matches: {n_hit}/{len(names)}", flush=True)
        rhos, traces = dms_for(circs, dict(zip(names, w)), 2 ** n_s)
        res["ae_init"] = analyse(rhos, A, B, C, f"n_s={n_s} ae_init")
    # L0-discard control
    c0, v0 = circuits_for(0, n_s)
    m0 = NumpyModel.from_diagrams(c0, use_jit=False)
    os.environ["QFM_WARMSTART"] = "fixed"
    w0v = exp13.warmstart_weights(m0)
    rhos0, _ = dms_for(c0, dict(zip([str(s) for s in m0.symbols], w0v)),
                       2 ** n_s)
    sq0 = sqrtm(rhos0[0])
    f0 = min(uhl(sq0, r) for r in rhos0[1:])
    res["L0_control_min_uhlmann"] = float(f0)
    print(f"[22] n_s={n_s} L0-discard control min fidelity = {f0:.12f}",
          flush=True)
    OUT[f"n_s_{n_s}"] = res
    json.dump(OUT, open("results_exp22.json", "w"), indent=2)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp22.json", "w"), indent=2)
print(f"[22] DONE in {OUT['runtime_sec']}s", flush=True)
