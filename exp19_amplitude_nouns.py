"""Exp 19: amplitude-encoded noun states — the first exact embedding->state
pipeline in this stack.

Anatomy (inspect_noun.log): every noun word-box compiles to Rx(_2) >> Rz(_1)
>> Rx(_0) >> Bra(0) on its own qubit — a universal Euler chain, so ANY
1-qubit effect is realizable exactly. We SOLVE the three angles per noun so
its word-box equals the amplitude encoding of its AraVec vector projected to
2 dims (PCA-2 over the dataset vocabulary, unit-normalized): fidelity between
two noun states then equals |cosine|^2 of their projected embeddings BY
CONSTRUCTION. Verbs (multi-leg boxes) are not exactly solvable at 1 layer —
two variants: V1 verbs = hash (isolates the noun effect), V2 verbs =
typed-PCA (exp18 encoder).

Evaluation: exp17/18 geometry protocol at n_s = 1 and 2.
Baselines to beat (n_s=2 AUC A-vs-B): hash 0.649 (param-identity effect),
index-sampled W2 0.569, typed-PCA 0.530.
"""
import json, time, math, hashlib
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
import sympy
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

try:
    from lambeq.backend.quantum import Ket, Bra, Rx, Rz
except ImportError:
    import lambeq.backend.quantum as Q
    Ket, Bra, Rx, Rz = Q.Ket, Q.Bra, Q.Rx, Q.Rz

RNG = np.random.default_rng(42)
SOLVE_RNG = np.random.default_rng(7)
WVEC = json.load(open("exp16_wordvecs.json"))["vectors"]

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
print(f"[19] twins: {len(twins)}", flush=True)

# ── noun targets: PCA-2 amplitude encoding ──────────────────────────────────
verbs, nouns = set(), set()
for d in data:
    toks = d["sentence"].split()
    if d["label"].endswith("VSO"):
        verbs.add(toks[0]); nouns.update(toks[1:])
    else:
        verbs.add(toks[1]); nouns.update([toks[0], toks[2]])
nlist = sorted(nouns)
mat = np.array([WVEC[w] for w in nlist])
S2 = PCA(n_components=2, random_state=0).fit_transform(mat)
targets = {}
for w, row in zip(nlist, S2):
    v = row / np.linalg.norm(row)
    targets[w] = np.array([v[0], v[1]], dtype=complex)
print(f"[19] {len(targets)} noun targets (PCA-2, unit norm)", flush=True)

# ── solver: <0| Rx(s2) Rz(s1) Rx(s0) == <target| up to global phase ────────
S2s, S1s, S0s = sympy.symbols("s2 s1 s0")
_amp = {}
for k in (0, 1):
    d = Ket(k) >> Rx(S2s) >> Rz(S1s) >> Rx(S0s) >> Bra(0)
    _amp[k] = d.lambdify(S2s, S1s, S0s)


def effect_vec(x):
    out = []
    for k in (0, 1):
        r = _amp[k](*x)
        try:
            r = r.eval()
        except AttributeError:
            pass
        out.append(complex(np.asarray(r).flatten()[0]))
    return np.array(out)


def solve_angles(target):
    def loss(x):
        v = effect_vec(x)
        n = np.linalg.norm(v)
        return 1.0 if n < 1e-9 else 1.0 - (abs(np.vdot(v, target)) / n) ** 2
    best = None
    for _ in range(8):
        r = minimize(loss, SOLVE_RNG.uniform(0, 2 * np.pi, 3),
                     method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
        if best.fun < 1e-10:
            break
    return best.x, float(best.fun)


solved, residuals = {}, []
for w in nlist:
    x, resid = solve_angles(targets[w])
    solved[w] = x
    residuals.append(resid)
print(f"[19] solved {len(solved)} nouns; max residual "
      f"{max(residuals):.2e}, median {np.median(residuals):.2e}", flush=True)

# ── typed-PCA for verbs (V2), from exp18 ────────────────────────────────────
def parse_symbol(name):
    word_part, rest = name.split("__", 1)
    base = word_part.split("_")[0]
    tstr, idx_s = rest.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        tstr, idx = rest, 0
    return base, tstr, idx


def typed_pca_rows(names):
    per_type, need = defaultdict(set), defaultdict(int)
    parsed = [parse_symbol(nm) for nm in names]
    for base, tstr, idx in parsed:
        per_type[tstr].add(base)
        need[tstr] = max(need[tstr], idx + 1)
    rows = {}
    for tstr, bases in per_type.items():
        have = sorted(b for b in bases if b in WVEC)
        if not have:
            continue
        vecs = np.array([WVEC[b] for b in have])
        k = int(min(need[tstr], max(1, len(have) - 1), vecs.shape[1]))
        pca = PCA(n_components=k, random_state=0)
        S = pca.fit_transform(vecs)
        lo, hi = S.min(axis=0), S.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        for b, row in zip(have, (S - lo) / span * 2 * np.pi):
            rows[(tstr, b)] = row
    return rows, parsed


def weights_for(names, variant):
    rows, parsed = typed_pca_rows(names)
    w = np.empty(len(names))
    n_ae = n_tp = n_hash = 0
    for i, (nm, (base, tstr, idx)) in enumerate(zip(names, parsed)):
        if tstr == "n" and base in solved and idx < 3:
            # chain is Rx(_2) >> Rz(_1) >> Rx(_0): solved x = (s2, s1, s0)
            w[i] = float(solved[base][{2: 0, 1: 1, 0: 2}[idx]]); n_ae += 1
        elif variant == "V2" and (tstr, base) in rows and idx < len(rows[(tstr, base)]):
            w[i] = float(rows[(tstr, base)][idx]); n_tp += 1
        else:
            h = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
            w[i] = (h / 0xFFFFFFFF) * 2 * math.pi; n_hash += 1
    print(f"[19] {variant}: {n_ae} amplitude-encoded, {n_tp} typed-PCA, "
          f"{n_hash} hash", flush=True)
    return w


# ── states + geometry (exp17/18 protocol) ───────────────────────────────────
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
        out[nm] = dict(auc=float(obs),
                       perm_p=float((np.sum(perm >= obs) + 1) / 10001),
                       mwu_p=float(mannwhitneyu(fA, neg,
                                                alternative="greater")[1]))
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
    print(f"[19] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC A-vs-B={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) | top1={out['retrieval']['top1']:.3f} "
          f"MRR={out['retrieval']['mrr']:.3f}", flush=True)
    return out


OUT = {"n_true_twins": len(twins),
       "solver": {"max_residual": float(max(residuals)),
                  "median_residual": float(np.median(residuals))},
       "baselines_ns2_auc": {"hash_W1": 0.649, "index_sampled_W2": 0.569,
                             "typed_pca": 0.530}}
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
    res = {}
    for variant in ("V1", "V2"):
        w = weights_for(names, variant)
        S, P = states_for(circs, w, names)
        res[variant] = analyse(S, A, B, C, f"n_s={n_s} {variant} "
                               f"(AE-nouns, {'hash' if variant=='V1' else 'typedPCA'}-verbs)")
        if n_s == 2 and variant == "V2":
            np.savez("states_L1_ns2_ampnouns.npz", states=S, norms=P,
                     vidx=vidx, labels=np.array([labels[i] for i in vidx]),
                     sentences=np.array([sents[i] for i in vidx]),
                     twin_pairs=np.array(A))
    OUT[f"n_s_{n_s}"] = res
    json.dump(OUT, open("results_exp19.json", "w"), indent=2)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp19.json", "w"), indent=2)
print(f"[19] DONE in {OUT['runtime_sec']}s", flush=True)
