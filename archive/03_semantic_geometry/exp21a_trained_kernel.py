"""Exp 21a: Lloyd-Schuld / kernel-target-alignment training THROUGH the
composition — the first trained quantum semantic space for Arabic.

Setup (evidence-driven): exp15c/17/18/19/20 proved no untrained circuit in
the lambeq ansatz family transports word similarity into sentence geometry.
This trains the word parameters against a sentence-level fidelity objective.

Design:
- Circuits: IQP, n_s_qubits=2 (the width where content anchoring first
  flickered), n_layers=1. 120 sentences, 60 true twins.
- Split by PAIR: 36 train / 12 val / 12 test (seed 42). Test pairs are never
  touched by the objective.
- Init: exp19 exact amplitude-encoded nouns + typed-PCA verbs (start at the
  provably-faithful point).
- Objective on train set: maximize mean fid(twins) - mean fid(same-order
  non-twins) with a small L2 pull to the init. Optimizer: SPSA.
- Model selection on val margin; final report = exp17 protocol (A/B/C AUC,
  10k permutations, retrieval) on the 12 held-out test pairs, pre vs post.
- Word-level check: spearman(noun-state fidelity, AraVec cosine) pre/post —
  does training preserve, rediscover, or destroy word-level semantics?

Speed: a custom numpy statevector interpreter replaces lambdify+eval per
call. It is CALIBRATED against lambeq itself: gate-matrix conventions are
tried in candidate sets and the interpreter must reproduce lambdify states
for validation circuits to 1e-9 fidelity BEFORE training starts, else abort.
"""
import json, time, math, hashlib
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from scipy.optimize import minimize
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

try:
    from lambeq import Symbol as LSymbol
except ImportError:
    from lambeq.backend.symbol import Symbol as LSymbol
from lambeq.backend.quantum import Ket, Bra, Rx, Rz

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
perm = RNG.permutation(len(twins))
tr_p = [twins[i] for i in perm[:36]]
va_p = [twins[i] for i in perm[36:48]]
te_p = [twins[i] for i in perm[48:60]]
print(f"[21a] twins {len(twins)} -> train {len(tr_p)} / val {len(va_p)} / "
      f"test {len(te_p)}", flush=True)

# ── circuits ────────────────────────────────────────────────────────────────
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
print(f"[21a] {len(circs)} circuits, {len(names)} symbols", flush=True)

# ── fast interpreter ────────────────────────────────────────────────────────
def compile_circuit(c):
    """-> list of ops: (kind, offset, sym_idx or const)."""
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
            val = complex(np.asarray(box.array).flatten()[0])
            prog.append(("scalar", 0, val))
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
            prog.append(("fixed2", off, nm if nm != "CNOT" else "CX"))
        elif nm in ("X", "Z") and nd == 1:
            prog.append(("fixed1", off, nm))
        else:
            raise ValueError(f"unknown box: {nm!r} dom={nd} cod={nc}")
    return prog

H_M = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
KETS = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
FIXED1 = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Z": np.diag([1.0 + 0j, -1.0])}
FIXED2 = {"CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
          "CZ": np.diag([1, 1, 1, -1]).astype(complex)}

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
            st = np.moveaxis(np.tensordot(FIXED1[arg], st, ([1], [off])),
                             0, off)
        elif kind == "fixed2":
            T = FIXED2[arg].reshape(2, 2, 2, 2)
            st = np.moveaxis(np.tensordot(T, st, ([2, 3], [off, off + 1])),
                             [0, 1], [off, off + 1])
        elif kind in ("Rx", "Rz"):
            U = forms[kind](w[arg])
            st = np.moveaxis(np.tensordot(U, st, ([1], [off])), 0, off)
        else:  # CRz / CRx on (off, off+1)
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
    ok = True
    for k in RNG.choice(len(circs), 5, replace=False):
        sf, pf = run_prog(progs[k], w_test, forms)
        sl, pl = lambdify_state(circs[k], w_test)
        if abs(abs(np.vdot(sf, sl)) ** 2 - 1) > 1e-9 or abs(pf - pl) > 1e-9:
            ok = False
            break
    if ok:
        FORMS = forms
        print(f"[21a] interpreter CALIBRATED (CRz variant {variant}); "
              f"5/5 circuits match lambdify to 1e-9", flush=True)
        break
if FORMS is None:
    raise SystemExit("[21a] ABORT: interpreter failed calibration")

def all_states(w, idxs):
    return {k: run_prog(progs[k], w, FORMS)[0] for k in idxs}

def fid(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)

# ── init weights: AE nouns + typed-PCA verbs (exp19 recipe) ────────────────
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
targets = {w: (r / np.linalg.norm(r)).astype(complex) for w, r in zip(nlist, S2)}

S2s, S1s, S0s = LSymbol("s2"), LSymbol("s1"), LSymbol("s0")
_amp = {k: (Ket(k) >> Rx(S2s) >> Rz(S1s) >> Rx(S0s) >> Bra(0)).lambdify(
    S2s, S1s, S0s) for k in (0, 1)}

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
    for _ in range(6):
        r = minimize(loss, SOLVE_RNG.uniform(0, 2 * np.pi, 3),
                     method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
        if best.fun < 1e-9:
            break
    return best.x

solved = {w: solve_angles(targets[w]) for w in nlist}
print(f"[21a] {len(solved)} noun AE solutions", flush=True)

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
per_type, need = defaultdict(set), defaultdict(int)
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

w0 = np.empty(len(names))
for i, (nm, (base, tstr, idx)) in enumerate(zip(names, parsed)):
    if tstr == "n" and base in solved and idx < 3:
        w0[i] = float(solved[base][{2: 0, 1: 1, 0: 2}[idx]])
    elif (tstr, base) in rows and idx < len(rows[(tstr, base)]):
        w0[i] = float(rows[(tstr, base)][idx])
    else:
        h = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
        w0[i] = (h / 0xFFFFFFFF) * 2 * math.pi

# ── objective ───────────────────────────────────────────────────────────────
def pair_sets(pairs):
    A = [(pos[i], pos[j]) for i, j in pairs if i in pos and j in pos]
    sent_k = sorted({k for ab in A for k in ab})
    svo_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("SVO")]
    vso_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("VSO")]
    rngl = np.random.default_rng(123)
    B, seen = [], set()
    npairs = min(60, len(svo_k) * (len(svo_k) - 1))
    for pool in (svo_k, vso_k):
        while len(B) < (npairs // 2 if pool is svo_k else npairs):
            i, j = rngl.choice(pool, 2, replace=False)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key); B.append((int(i), int(j)))
    return A, B, sent_k

A_tr, B_tr, k_tr = pair_sets(tr_p)
A_va, B_va, k_va = pair_sets(va_p)

def margin(w, A, B, idxs):
    S = all_states(w, idxs)
    fA = np.mean([fid(S[i], S[j]) for i, j in A])
    fB = np.mean([fid(S[i], S[j]) for i, j in B])
    return fA - fB

LAMB = 1e-3
def loss(w):
    return -margin(w, A_tr, B_tr, k_tr) + LAMB * float(np.mean((w - w0) ** 2))

t_l = time.time()
_ = loss(w0)
per_eval = time.time() - t_l
print(f"[21a] loss eval: {per_eval*1000:.0f} ms "
      f"(init train margin {margin(w0, A_tr, B_tr, k_tr):+.4f}, "
      f"val {margin(w0, A_va, B_va, k_va):+.4f})", flush=True)

ITERS = 8000
w = w0.copy()
best_val, best_w = -np.inf, w0.copy()
history = []
srng = np.random.default_rng(0)
for k in range(ITERS):
    ak = 0.25 / (k + 50) ** 0.602
    ck = 0.12 / (k + 1) ** 0.101
    d = srng.choice([-1.0, 1.0], size=len(w))
    g = (loss(w + ck * d) - loss(w - ck * d)) / (2 * ck) * d
    w = w - ak * g
    if (k + 1) % 200 == 0:
        m_tr = margin(w, A_tr, B_tr, k_tr)
        m_va = margin(w, A_va, B_va, k_va)
        history.append({"iter": k + 1, "train_margin": m_tr,
                        "val_margin": m_va})
        if m_va > best_val:
            best_val, best_w = m_va, w.copy()
        print(f"[21a] it {k+1}: train {m_tr:+.4f} val {m_va:+.4f} "
              f"(best val {best_val:+.4f})", flush=True)

# ── evaluation on held-out test pairs ───────────────────────────────────────
def analyse(w, pairs, tag):
    A, B, idxs = pair_sets(pairs)
    sent_k = idxs
    svo_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("SVO")]
    vso_k = [k for k in sent_k if str(labels[vidx[k]]).endswith("VSO")]
    twinset = set((min(a, b), max(a, b)) for a, b in A)
    rngl = np.random.default_rng(321)
    C, seen = [], set()
    while len(C) < len(A) * 2 and len(seen) < 500:
        i = int(rngl.choice(svo_k)); j = int(rngl.choice(vso_k))
        key = (min(i, j), max(i, j))
        seen.add(key)
        if key in twinset or (i, j) in C:
            continue
        C.append((i, j))
    S = all_states(w, sent_k)
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
    print(f"[21a] {tag}: A={out['medians']['A']:.4f} "
          f"B={out['medians']['B']:.4f} C={out['medians']['C']:.4f} | "
          f"AUC A-vs-B={out['A_vs_B']['auc']:.4f} "
          f"(p={out['A_vs_B']['perm_p']:.5f})", flush=True)
    return out

def noun_semantics(w, tag):
    vecs, cos, fids = {}, [], []
    for wd in nlist:
        idxs3 = {}
        for i, (nm, (base, tstr, idx)) in enumerate(zip(names, parsed)):
            if base == wd and tstr == "n" and idx < 3:
                idxs3[idx] = w[i]
        if len(idxs3) == 3:
            vecs[wd] = effect_vec([idxs3[2], idxs3[1], idxs3[0]])
    keys = sorted(vecs)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            va, vb = np.array(WVEC[keys[a]]), np.array(WVEC[keys[b]])
            cos.append(float(np.dot(va, vb) /
                             (np.linalg.norm(va) * np.linalg.norm(vb))))
            fids.append(fid(vecs[keys[a]] / np.linalg.norm(vecs[keys[a]]),
                            vecs[keys[b]] / np.linalg.norm(vecs[keys[b]])))
    rho, p = spearmanr(cos, fids)
    print(f"[21a] noun semantics {tag}: spearman(fid, cos)={rho:.4f} "
          f"(p={p:.2e}, n={len(cos)})", flush=True)
    return {"spearman": float(rho), "p": float(p), "n": len(cos)}

OUT = {"split": {"train": 36, "val": 12, "test": 12},
       "calibration": "passed", "loss_eval_ms": per_eval * 1000,
       "iters": ITERS, "history": history}
OUT["pre"] = {"test": analyse(w0, te_p, "PRE test"),
              "train": analyse(w0, tr_p, "PRE train"),
              "nouns": noun_semantics(w0, "PRE")}
OUT["post"] = {"test": analyse(best_w, te_p, "POST test"),
               "train": analyse(best_w, tr_p, "POST train"),
               "val_margin_best": best_val,
               "nouns": noun_semantics(best_w, "POST")}
np.savez("exp21a_weights.npz", w0=w0, w_trained=best_w,
         names=np.array(names))
json.dump(OUT, open("results_exp21a.json", "w"), indent=2)
OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp21a.json", "w"), indent=2)
print(f"[21a] DONE in {OUT['runtime_sec']}s", flush=True)
