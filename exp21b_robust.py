"""Exp 21b-lite: tiny constrained lexical encoder, real data only.

THE CLAIM UNDER TEST (the program's actual thesis, now falsifiable):
composition fixed by grammar (zero learned parameters in how words combine)
+ a lexical encoder measured in dozens of parameters + 36 real training
pairs => semantic geometry that GENERALIZES to held-out pairs, while
preserving order-discrimination. Neither classical extreme offers this
jointly: bag-of-vectors is meaning-perfect but order-blind by construction
on matched twins; order-sensitive classical models at the same budget are
the matched control (C2) trained in this same script.

Encoder: per grammatical type t, fixed unsupervised PCA of that type's
AraVec vectors to d=6 dims (z-scored), then a learned linear map
W_t (k_t x 6) + b_t -> circuit parameters. Initialized by least squares to
reproduce exp21a's faithful init (AE nouns + typed-PCA verbs) within the
constrained class. Optimizer: SPSA on ~100-250 encoder weights.
Split identical to exp21a: 36 train / 12 val / 12 test pairs (seed 42).

Pre-registered outcomes:
- Generalization: post-training test AUC(A vs B) > 0.5 with perm p < 0.05.
- Order preservation: SVO/VSO test accuracy from output statistics does not
  collapse post-training.
- Pareto vs C1 (bag-of-vectors: meaning AUC 1.0 trivially, order = chance)
  and C2 (matched-budget classical metric on order-sensitive concatenation).
Failure routes (pre-registered): R1 val flat -> widen d / relax class;
R2 train up test flat -> inductive bias wrong, go to transport-theory track;
R3 order collapses -> add order-preservation term; R4 C2 wins both axes ->
report honestly, quantum case rests on structural/analog/hardware results.
"""
import json, time, math, hashlib, os
import numpy as np
from collections import defaultdict

SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from lambeq import Symbol as LSymbol
except ImportError:
    from lambeq.backend.symbol import Symbol as LSymbol
from lambeq.backend.quantum import Ket, Bra, Rx, Rz

RNG = np.random.default_rng(SPLIT_SEED)
print(f"[21b] SPLIT_SEED = {SPLIT_SEED}", flush=True)
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
perm = RNG.permutation(len(twins))
tr_p = [twins[i] for i in perm[:36]]
va_p = [twins[i] for i in perm[36:48]]
te_p = [twins[i] for i in perm[48:60]]

N_S = 2
diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)
ansatz = exp13.make_ansatz(1, N_S)
circs, vidx = [], []
for i, d in enumerate(diagrams):
    if d is None:
        continue
    try:
        circs.append(ansatz(exp13._remove_cups(d)))
        vidx.append(i)
    except Exception:
        pass
pos = {orig: k for k, orig in enumerate(vidx)}
model = NumpyModel.from_diagrams(circs, use_jit=False)
names = [str(s) for s in model.symbols]
sym_index = {nm: i for i, nm in enumerate(names)}

npz = np.load("exp21a_weights.npz", allow_pickle=True)
assert list(npz["names"]) == names, "symbol order mismatch vs exp21a"
w0 = npz["w0"].astype(float)
print(f"[21b] {len(circs)} circuits, {len(names)} symbols, w0 loaded",
      flush=True)

# ── fast interpreter (calibrated, from exp21a incl. CX fix) ────────────────
H_M = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
KETS = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
FIXED1 = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.diag([1.0 + 0j, -1.0])}
FIXED2 = {"CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
          "CZ": np.diag([1, 1, 1, -1]).astype(complex)}

def compile_circuit(c):
    try:
        pairs = list(zip(c.boxes, c.offsets))
    except AttributeError:
        pairs = [(lay.box, len(lay.left)) for lay in c.layers]
    prog = []
    for box, off in pairs:
        nm = str(getattr(box, "name", box))
        fs = list(getattr(box, "free_symbols", []))
        si = sym_index[str(fs[0])] if fs else None
        nd, nc = len(box.dom), len(box.cod)
        if nd == 0 and nc == 1:
            prog.append(("ket", off, int(nm) if nm in "01" else 0))
        elif nd == 1 and nc == 0:
            prog.append(("bra", off, int(nm) if nm in "01" else 0))
        elif nd == 0 and nc == 0:
            prog.append(("scalar", 0, complex(np.asarray(box.array).flatten()[0])))
        elif nm == "H":
            prog.append(("H", off, None))
        elif nm.startswith("Rx("):
            prog.append(("Rx", off, si))
        elif nm.startswith("Rz("):
            prog.append(("Rz", off, si))
        elif nm.startswith("CRz("):
            prog.append(("CRz", off, si))
        elif nm.startswith("CRx("):
            prog.append(("CRx", off, si))
        elif nm.upper().startswith("SWAP"):
            prog.append(("SWAP", off, None))
        elif nm in ("CX", "CNOT", "CZ") and nd == 2:
            prog.append(("fixed2", off, "CX" if nm != "CZ" else "CZ"))
        elif nm in ("X", "Z") and nd == 1:
            prog.append(("fixed1", off, nm))
        else:
            raise ValueError(f"unknown box: {nm!r} dom={nd} cod={nc}")
    return prog

def gate_forms(variant):
    def rx(t):
        c, s = np.cos(np.pi * t), np.sin(np.pi * t)
        return np.array([[c, -1j * s], [-1j * s, c]])
    def rz(t):
        return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])
    if variant == 0:
        crz = lambda t: np.diag([1, 1, np.exp(-1j * np.pi * t),
                                 np.exp(1j * np.pi * t)])
    else:
        crz = lambda t: np.diag([1, 1, 1, np.exp(2j * np.pi * t)])
    crx = lambda t: np.block([[np.eye(2), np.zeros((2, 2))],
                              [np.zeros((2, 2)), rx(t)]])
    return {"Rx": rx, "Rz": rz, "CRz": crz, "CRx": crx}

def run_prog(prog, w, forms):
    st = np.array(1.0 + 0j)
    for kind, off, arg in prog:
        if kind == "ket":
            st = np.moveaxis(np.tensordot(st, KETS[arg], 0), -1, off)
        elif kind == "bra":
            st = np.take(st, arg, axis=off)
        elif kind == "scalar":
            st = st * arg
        elif kind == "H":
            st = np.moveaxis(np.tensordot(H_M, st, ([1], [off])), 0, off)
        elif kind == "SWAP":
            st = np.swapaxes(st, off, off + 1)
        elif kind == "fixed1":
            st = np.moveaxis(np.tensordot(FIXED1[arg], st, ([1], [off])), 0, off)
        elif kind == "fixed2":
            T = FIXED2[arg].reshape(2, 2, 2, 2)
            st = np.moveaxis(np.tensordot(T, st, ([2, 3], [off, off + 1])),
                             [0, 1], [off, off + 1])
        elif kind in ("Rx", "Rz"):
            U = forms[kind](w[arg])
            st = np.moveaxis(np.tensordot(U, st, ([1], [off])), 0, off)
        else:
            T = forms[kind](w[arg]).reshape(2, 2, 2, 2)
            st = np.moveaxis(np.tensordot(T, st, ([2, 3], [off, off + 1])),
                             [0, 1], [off, off + 1])
    amps = np.asarray(st).flatten()
    p = float(np.sum(np.abs(amps) ** 2))
    return (amps / np.sqrt(p) if p > 1e-14 else amps * np.nan), p

progs = [compile_circuit(c) for c in circs]

def lambdify_state(c, w):
    syms = sorted(c.free_symbols, key=str)
    vals = [w[sym_index[str(s)]] for s in syms]
    amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
    p = float(np.sum(np.abs(amps) ** 2))
    return amps / np.sqrt(p), p

w_test = RNG.uniform(0, 2 * np.pi, len(names))
FORMS = None
for variant in (0, 1):
    forms = gate_forms(variant)
    if all(abs(abs(np.vdot(run_prog(progs[k], w_test, forms)[0],
                           lambdify_state(circs[k], w_test)[0])) ** 2 - 1) < 1e-9
           for k in RNG.choice(len(circs), 5, replace=False)):
        FORMS = forms
        print(f"[21b] interpreter calibrated (variant {variant})", flush=True)
        break
if FORMS is None:
    raise SystemExit("[21b] ABORT: calibration failed")

def fid(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)

# ── constrained encoder ─────────────────────────────────────────────────────
def parse_symbol(name):
    word_part, rest = name.split("__", 1)
    base = word_part.split("_")[0]
    tstr, idx_s = rest.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        tstr, idx = rest, 0
    return base, tstr, idx

parsed = [parse_symbol(nm) for nm in names]
D_IN = 6
types = {}
n_frozen = 0
for base, tstr, idx in parsed:
    if base not in WVEC:
        n_frozen += 1
        continue
    t = types.setdefault(tstr, {"words": set(), "k": 0})
    t["words"].add(base)
    t["k"] = max(t["k"], idx + 1)
print(f"[21b] {n_frozen} non-word symbols frozen at init values", flush=True)
Z = {}
for tstr, t in types.items():
    words = sorted(t["words"])
    mat = np.array([WVEC[w] for w in words])
    d = min(D_IN, len(words) - 1, mat.shape[1])
    z = PCA(n_components=d, random_state=0).fit_transform(mat)
    z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1.0)
    if d < D_IN:
        z = np.hstack([z, np.zeros((len(words), D_IN - d))])
    Z[tstr] = {"words": words, "row": {w: i for i, w in enumerate(words)},
               "z": z}

# encoder init: least-squares fit to reproduce w0 within the class
type_list = sorted(types)
n_params = 0
inits = {}
for tstr in type_list:
    k = types[tstr]["k"]
    zt = Z[tstr]["z"]
    T0 = np.zeros((len(Z[tstr]["words"]), k))
    cnt = np.zeros((len(Z[tstr]["words"]), k))
    for i, (base, ts, idx) in enumerate(parsed):
        if ts == tstr and base in Z[tstr]["row"]:
            T0[Z[tstr]["row"][base], idx] = w0[i]
            cnt[Z[tstr]["row"][base], idx] = 1
    Xd = np.hstack([zt, np.ones((zt.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(Xd, T0, rcond=None)
    inits[tstr] = coef            # (D_IN+1, k)
    n_params += coef.size
print(f"[21b] encoder classes: " +
      ", ".join(f"{t}:k={types[t]['k']},n={len(types[t]['words'])}"
                for t in type_list), flush=True)
print(f"[21b] TOTAL LEARNED PARAMETERS: {n_params}", flush=True)

def pack(coefs):
    return np.concatenate([coefs[t].ravel() for t in type_list])

def unpack(theta):
    out, o = {}, 0
    for tstr in type_list:
        k = types[tstr]["k"]
        n = (D_IN + 1) * k
        out[tstr] = theta[o:o + n].reshape(D_IN + 1, k)
        o += n
    return out

theta0 = pack(inits)

def weights_from(theta):
    coefs = unpack(theta)
    w = w0.copy()   # non-word symbols stay frozen at init
    per_type_params = {}
    for tstr in type_list:
        zt = Z[tstr]["z"]
        Xd = np.hstack([zt, np.ones((zt.shape[0], 1))])
        per_type_params[tstr] = Xd @ coefs[tstr]
    for i, (base, tstr, idx) in enumerate(parsed):
        if tstr in Z and base in Z[tstr]["row"] and idx < types[tstr]["k"]:
            w[i] = per_type_params[tstr][Z[tstr]["row"][base], idx]
    return w

# ── pairs, loss, training ───────────────────────────────────────────────────
def pair_sets(pairs, seed=123):
    A = [(pos[i], pos[j]) for i, j in pairs if i in pos and j in pos]
    sent_k = sorted({k for ab in A for k in ab})
    svo_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("SVO")]
    vso_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("VSO")]
    rngl = np.random.default_rng(seed)
    B, seen = [], set()
    target = min(60, len(svo_k) * (len(svo_k) - 1) // 2 * 2)
    for pool in (svo_k, vso_k):
        goal = target // 2 if pool is svo_k else target
        while len(B) < goal:
            i, j = rngl.choice(pool, 2, replace=False)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            B.append((int(i), int(j)))
    return A, B, sent_k

A_tr, B_tr, k_tr = pair_sets(tr_p)
A_va, B_va, k_va = pair_sets(va_p)

def states(w, idxs):
    return {k: run_prog(progs[k], w, FORMS)[0] for k in idxs}

def margin(w, A, B, idxs):
    S = states(w, idxs)
    return (np.mean([fid(S[i], S[j]) for i, j in A]) -
            np.mean([fid(S[i], S[j]) for i, j in B]))

LAMB = 1e-4
def loss(theta):
    w = weights_from(theta)
    return -margin(w, A_tr, B_tr, k_tr) + LAMB * float(
        np.mean((theta - theta0) ** 2))

m0_tr = margin(weights_from(theta0), A_tr, B_tr, k_tr)
m0_va = margin(weights_from(theta0), A_va, B_va, k_va)
print(f"[21b] encoder-init margins: train {m0_tr:+.4f} val {m0_va:+.4f} "
      f"(raw-w0 train {margin(w0, A_tr, B_tr, k_tr):+.4f})", flush=True)

ITERS = 6000
theta = theta0.copy()
best_val, best_theta = -np.inf, theta0.copy()
history = []
srng = np.random.default_rng(0)
for k in range(ITERS):
    ak = 0.08 / (k + 50) ** 0.602
    ck = 0.05 / (k + 1) ** 0.101
    dvec = srng.choice([-1.0, 1.0], size=len(theta))
    g = (loss(theta + ck * dvec) - loss(theta - ck * dvec)) / (2 * ck) * dvec
    theta = theta - ak * g
    if (k + 1) % 200 == 0:
        w = weights_from(theta)
        m_tr = margin(w, A_tr, B_tr, k_tr)
        m_va = margin(w, A_va, B_va, k_va)
        history.append({"iter": k + 1, "train": m_tr, "val": m_va})
        if m_va > best_val:
            best_val, best_theta = m_va, theta.copy()
        print(f"[21b] it {k+1}: train {m_tr:+.4f} val {m_va:+.4f} "
              f"(best val {best_val:+.4f})", flush=True)

# ── evaluation ──────────────────────────────────────────────────────────────
def analyse(w, pairs, tag):
    A, B, idxs = pair_sets(pairs)
    svo_k = [k for k in idxs if str(labels[vidx[k]]).endswith("SVO")]
    vso_k = [k for k in idxs if str(labels[vidx[k]]).endswith("VSO")]
    twinset = set((min(a, b), max(a, b)) for a, b in A)
    rngl = np.random.default_rng(321)
    C, seen = [], set()
    while len(C) < len(A) * 2 and len(seen) < 500:
        i = int(rngl.choice(svo_k))
        j = int(rngl.choice(vso_k))
        key = (min(i, j), max(i, j))
        seen.add(key)
        if key in twinset or (i, j) in C:
            continue
        C.append((i, j))
    S = states(w, idxs)
    fA = np.array([fid(S[i], S[j]) for i, j in A])
    fB = np.array([fid(S[i], S[j]) for i, j in B])
    fC = np.array([fid(S[i], S[j]) for i, j in C])
    out = {"medians": {"A": float(np.median(fA)), "B": float(np.median(fB)),
                       "C": float(np.median(fC))}}
    for nm, neg in (("A_vs_B", fB), ("A_vs_C", fC)):
        y = np.r_[np.ones(len(fA)), np.zeros(len(neg))]
        s = np.r_[fA, neg]
        obs = roc_auc_score(y, s)
        prm = np.array([roc_auc_score(RNG.permutation(y), s)
                        for _ in range(10000)])
        out[nm] = {"auc": float(obs),
                   "perm_p": float((np.sum(prm >= obs) + 1) / 10001)}
    print(f"[21b] {tag}: A={out['medians']['A']:.4f} B={out['medians']['B']:.4f} "
          f"C={out['medians']['C']:.4f} | AUC_AB={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f}) AUC_AC={out['A_vs_C']['auc']:.4f}",
          flush=True)
    return out

def order_eval(w, tag):
    from sklearn.model_selection import StratifiedKFold
    X, y = [], []
    for k in range(len(progs)):
        amps, p = run_prog(progs[k], w, FORMS)
        X.append(list(np.abs(amps) ** 2 * p) + [p])
        y.append(1 if str(labels[vidx[k]]).endswith("SVO") else 0)
    X, y = np.array(X), np.array(y)
    accs = []
    for seed in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))])
            pipe.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], pipe.predict(X[te])))
    acc = float(np.mean(accs))
    print(f"[21b] order {tag}: CV acc = {acc:.4f} +/- {np.std(accs):.4f}",
          flush=True)
    return acc

_amp = {k: (Ket(k) >> Rx(LSymbol("s2")) >> Rz(LSymbol("s1")) >>
            Rx(LSymbol("s0")) >> Bra(0)).lambdify(
    LSymbol("s2"), LSymbol("s1"), LSymbol("s0")) for k in (0, 1)}

def noun_semantics(w, tag):
    vecs = {}
    for wd in sorted(types["n"]["words"]) if "n" in types else []:
        idxs3 = {}
        for i, (base, tstr, idx) in enumerate(parsed):
            if base == wd and tstr == "n" and idx < 3:
                idxs3[idx] = w[i]
        if len(idxs3) == 3:
            out = []
            for kk in (0, 1):
                r = _amp[kk](idxs3[2], idxs3[1], idxs3[0])
                try:
                    r = r.eval()
                except AttributeError:
                    pass
                out.append(complex(np.asarray(r).flatten()[0]))
            v = np.array(out)
            vecs[wd] = v / np.linalg.norm(v)
    keys = sorted(vecs)
    cos, fids = [], []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            va, vb = np.array(WVEC[keys[a]]), np.array(WVEC[keys[b]])
            cos.append(float(np.dot(va, vb) /
                             (np.linalg.norm(va) * np.linalg.norm(vb))))
            fids.append(fid(vecs[keys[a]], vecs[keys[b]]))
    rho, p = spearmanr(cos, fids)
    print(f"[21b] noun semantics {tag}: rho={rho:.4f} (p={p:.1e})", flush=True)
    return {"spearman": float(rho), "p": float(p)}

# classical controls
def sent_z(orig):
    toks = sents[orig].split()
    out = []
    for w_ in toks:
        row = None
        for cand in type_list:
            if w_ in Z[cand]["row"]:
                row = Z[cand]["z"][Z[cand]["row"][w_]]
                break
        if row is None:
            print(f"[21b] WARN: token {w_!r} not in encoder vocab; zero-vec",
                  flush=True)
            row = np.zeros(D_IN)
        out.append(row)
    return out

def classical_controls():
    res = {}
    # C1 bag-of-vectors: twins identical => meaning AUC 1.0, order = chance
    A, B, idxs = pair_sets(te_p)
    bow = {k: np.mean(sent_z(vidx[k]), axis=0) for k in idxs}
    cosv = lambda a, b: float(np.dot(a, b) /
                              (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    sA = [cosv(bow[i], bow[j]) for i, j in A]
    sB = [cosv(bow[i], bow[j]) for i, j in B]
    y = np.r_[np.ones(len(sA)), np.zeros(len(sB))]
    res["C1_bow"] = {"meaning_auc": float(roc_auc_score(y, np.r_[sA, sB])),
                     "order_acc": 0.5,
                     "note": "twins are identical vectors: order-blind by "
                             "construction on matched pairs"}
    # C2 matched-budget metric on order-sensitive concatenation
    def feats_concat(k):
        zs = sent_z(vidx[k])[:3]          # first 3 tokens (S/V/O core);
        while len(zs) < 3:                # some sentences carry a 4th adjunct
            zs.append(np.zeros(D_IN))
        return np.concatenate(zs)
    M0 = np.zeros((8, 18))
    M0[:6, :6] = np.eye(6)
    thc = M0.ravel()
    A_t, B_t, k_t2 = pair_sets(tr_p)
    def c2_margin(th, A_, B_):
        M = th.reshape(8, 18)
        f = {k: M @ feats_concat(k) for k in
             sorted({x for ab in A_ + B_ for x in ab})}
        return (np.mean([cosv(f[i], f[j]) for i, j in A_]) -
                np.mean([cosv(f[i], f[j]) for i, j in B_]))
    def c2_loss(th):
        return -c2_margin(th, A_t, B_t) + 1e-4 * float(np.mean((th - thc) ** 2))
    th = thc.copy()
    rngc = np.random.default_rng(1)
    bv, bth = -np.inf, thc.copy()
    A_v, B_v, _ = pair_sets(va_p)
    for k in range(3000):
        ak = 0.08 / (k + 50) ** 0.602
        ck = 0.05 / (k + 1) ** 0.101
        dv = rngc.choice([-1.0, 1.0], size=len(th))
        g = (c2_loss(th + ck * dv) - c2_loss(th - ck * dv)) / (2 * ck) * dv
        th = th - ak * g
        if (k + 1) % 200 == 0:
            mv = c2_margin(th, A_v, B_v)
            if mv > bv:
                bv, bth = mv, th.copy()
    M = bth.reshape(8, 18)
    A_e, B_e, _ = pair_sets(te_p)
    f = {k: M @ feats_concat(k) for k in
         sorted({x for ab in A_e + B_e for x in ab})}
    sA = [cosv(f[i], f[j]) for i, j in A_e]
    sB = [cosv(f[i], f[j]) for i, j in B_e]
    y = np.r_[np.ones(len(sA)), np.zeros(len(sB))]
    res["C2_matched_classical"] = {
        "params": int(bth.size), "val_margin": float(bv),
        "meaning_auc_test": float(roc_auc_score(y, np.r_[sA, sB]))}
    print(f"[21b] controls: C1 meaning AUC {res['C1_bow']['meaning_auc']:.3f} "
          f"(order-blind); C2 test AUC "
          f"{res['C2_matched_classical']['meaning_auc_test']:.3f} "
          f"({bth.size} params)", flush=True)
    return res

w_init = weights_from(theta0)
w_best = weights_from(best_theta)
OUT = {"n_learned_params": int(n_params), "iters": ITERS,
       "history": history, "init_margins": {"train": m0_tr, "val": m0_va}}
OUT["pre"] = {"test": analyse(w_init, te_p, "PRE test"),
              "train": analyse(w_init, tr_p, "PRE train"),
              "order_test_acc": order_eval(w_init, "PRE"),
              "nouns": noun_semantics(w_init, "PRE")}
OUT["post"] = {"test": analyse(w_best, te_p, "POST test"),
               "train": analyse(w_best, tr_p, "POST train"),
               "order_test_acc": order_eval(w_best, "POST"),
               "nouns": noun_semantics(w_best, "POST"),
               "best_val_margin": float(best_val)}
OUT["split_seed"] = SPLIT_SEED
np.savez(f"exp21b_encoder_seed{SPLIT_SEED}.npz", theta0=theta0,
         theta_best=best_theta, names=np.array(names))
json.dump(OUT, open(f"results_exp21b_seed{SPLIT_SEED}.json", "w"), indent=2)
print("[21b] results + weights SAVED (pre-controls)", flush=True)
try:
    OUT["classical_controls"] = classical_controls()
except Exception as e:
    OUT["classical_controls"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[21b] controls failed: {e}", flush=True)
OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open(f"results_exp21b_seed{SPLIT_SEED}.json", "w"), indent=2)
print(f"[21b] DONE in {OUT['runtime_sec']}s", flush=True)
