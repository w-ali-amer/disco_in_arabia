"""Exp 20: does a different compositional ansatz transport word similarity?

exp15c/17/18/19 established: with the IQP ansatz, no input-side encoder
(index-sampled, tied, typed-PCA, exact amplitude encoding) produces robust
twin-proximity geometry. Before concluding the transport failure is generic,
sweep the ansatz family: StronglyEntanglingAnsatz, Sim14Ansatz, Sim15Ansatz
(SpiderAnsatz is a classical tensor ansatz — excluded from the quantum claim).

Protocol: exp17/18/19 geometry (true twins, A/B/C, AUC + 10k permutations,
retrieval) at n_s in {1, 2}, n_layers=1, weights in {hash, typed-PCA}
(content-free vs content-bearing probes; AE-noun solving is IQP-specific).
"""
import json, time, math, hashlib
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

ANSATZE = {}
try:
    from lambeq import StronglyEntanglingAnsatz
    ANSATZE["StronglyEntangling"] = StronglyEntanglingAnsatz
except ImportError as e:
    print("[20] StronglyEntangling unavailable:", e, flush=True)
try:
    from lambeq import Sim14Ansatz, Sim15Ansatz
    ANSATZE["Sim14"] = Sim14Ansatz
    ANSATZE["Sim15"] = Sim15Ansatz
except ImportError as e:
    print("[20] Sim14/15 unavailable:", e, flush=True)
print("[20] ansatze:", list(ANSATZE), flush=True)

RNG = np.random.default_rng(42)
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
print(f"[20] twins: {len(twins)}", flush=True)

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)


def parse_symbol(name):
    word_part, rest = name.split("__", 1)
    base = word_part.split("_")[0]
    tstr, idx_s = rest.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        tstr, idx = rest, 0
    return base, tstr, idx


def weights_hash(names):
    w = np.empty(len(names))
    for i, nm in enumerate(names):
        h = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
        w[i] = (h / 0xFFFFFFFF) * 2 * math.pi
    return w


def weights_typed_pca(names):
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
    w = np.empty(len(names))
    for i, (nm, (base, tstr, idx)) in enumerate(zip(names, parsed)):
        if (tstr, base) in rows and idx < len(rows[(tstr, base)]):
            w[i] = float(rows[(tstr, base)][idx])
        else:
            h = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
            w[i] = (h / 0xFFFFFFFF) * 2 * math.pi
    return w


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
    print(f"[20] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC A-vs-B={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) | top1={out['retrieval']['top1']:.3f}",
          flush=True)
    return out


OUT = {"n_true_twins": len(twins),
       "iqp_baselines_auc": {"ns1_hash": 0.580, "ns2_hash": 0.649,
                             "ns2_typed_pca": 0.530, "ns2_ae_nouns": 0.555}}
for aname, acls in ANSATZE.items():
    for n_s in (1, 2):
        try:
            ansatz = acls({exp13.S_ty: n_s, exp13.N_ty: 1}, n_layers=1)
        except Exception as e:
            print(f"[20] {aname} n_s={n_s} ansatz init failed: {e}", flush=True)
            continue
        circs, vidx = [], []
        for i, d in enumerate(diagrams):
            if d is None:
                continue
            try:
                circs.append(ansatz(exp13._remove_cups(d)))
                vidx.append(i)
            except Exception:
                pass
        if len(circs) < 100:
            print(f"[20] {aname} n_s={n_s}: only {len(circs)} circuits — skip",
                  flush=True)
            continue
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
        try:
            model = NumpyModel.from_diagrams(circs, use_jit=False)
        except Exception as e:
            print(f"[20] {aname} n_s={n_s} model failed: {e}", flush=True)
            continue
        names = [str(s) for s in model.symbols]
        res = {}
        for wname, wfn in (("hash", weights_hash),
                           ("typed_pca", weights_typed_pca)):
            try:
                S, P = states_for(circs, wfn(names), names)
                res[wname] = analyse(S, A, B, C,
                                     f"{aname} n_s={n_s} {wname}")
            except Exception as e:
                print(f"[20] {aname} n_s={n_s} {wname} failed: {e}", flush=True)
        OUT[f"{aname}_ns{n_s}"] = res
        json.dump(OUT, open("results_exp20.json", "w"), indent=2)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp20.json", "w"), indent=2)
print(f"[20] DONE in {OUT['runtime_sec']}s", flush=True)
