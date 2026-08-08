# -*- coding: utf-8 -*-
"""Exp 34a: zero-training structural plausibility — argument-swap
discrimination on real parses.

HONESTY NOTE (pre-registered): this is a CAPABILITY test, not a
computational-advantage test.  The zx4 encoder's solve targets are
classical statistics (AraVec centroids), so an information-matched
structured classical baseline is expected to TIE it.  The reportable
outcomes are: (a) the circuit discriminates real from argument-swapped
sentences with no training, through real (fusion-repaired) parses, at
parity with structure-aware classical scoring; (b) the bag-of-vectors
baseline is 50% BY CONSTRUCTION — both orders present the identical
input — which is the structural point, and it belongs to classical
models without syntax, not to classical models per se.

Design: for each verb's usable 3-token frames, build the argument-swapped
sentence (verb obj subj).  Per held-out frame i: fit zx4 theta on the
other frames (preference excludes frame i's subject), then score each
sentence by feeding the ACTUAL argument occupying the subject position
into the subject wire: score = |<evidence_row, enc(vec(argument))>|^2.
Correct if score(orig) > score(swap).  Control column (v1 finding, kept
deliberately): the filter-only score — evidence direction vs preference,
never touching the actual argument — is argument-blind by construction
and must sit at ~50% (measured 27/54 in the first run); it operationally
confirms the zx4 family's frame invariance.  Classical structured
baseline: same comparison with |<enc(word), pref>|^2 directly.

Run with ARABIC_POS_FUSION=1.
"""
import os, json, math
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

S_OUT = int(os.environ.get("S_OUT", "0"))
if os.environ.get("ARABIC_POS_FUSION", "0") != "1":
    print("[34a] WARNING: fusion off — frame supply will be starved",
          flush=True)
MAX_TRY_PER_VERB = 24
MAX_USE_PER_VERB = 10
MIN_FRAMES = 4

src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS = ns["KETS"], ns["FORMS"]
apply1, apply2 = ns["apply1"], ns["apply2"]

data_all = json.load(open("sentences.json", encoding="utf-8"))

def hn(w):
    return w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

frames = defaultdict(list)
subjects = defaultdict(list)
objects_pool = []
surface = {}
for split, ds in data_all.items():
    for d in ds:
        t = d["sentence"].split()
        lab = d.get("label", "")
        if len(t) != 3:          # swap needs clean V S O triples
            continue
        if lab.endswith("_VSO"):
            k = hn(t[0])
            surface.setdefault(k, t[0])
            frames[k].append((d["sentence"], t[1], t[2], split))
            subjects[k].append(t[1])
            objects_pool.append(t[2])
        elif lab.startswith("WSD_") and hn(t[1]) == hn(lab.split("_")[1]):
            k = hn(t[1])
            surface.setdefault(k, t[1])
            vso = " ".join([t[1], t[0], t[2]])
            frames[k].append((vso, t[0], t[2], split))
            subjects[k].append(t[0])
            objects_pool.append(t[2])
        elif lab.endswith("_SVO"):
            subjects[hn(t[1])].append(t[0])

def vec(wd):
    cands = [wd]
    if wd.startswith("ال"):
        cands.append(wd[2:])
    cands += [c.replace("ة", "ه").replace("ى", "ي") for c in list(cands)]
    for c in dict.fromkeys(cands):
        v = exp13._aravec_vec(c)
        if v is not None:
            return np.asarray(v, float)
    return None

anim_words = sorted({w for k in frames for w in subjects[k]})
inan_words = sorted(set(objects_pool))
anim_vecs = [v for v in (vec(w) for w in anim_words) if v is not None]
inan_vecs = [v for v in (vec(w) for w in inan_words) if v is not None]
axis = np.mean(anim_vecs, axis=0) - np.mean(inan_vecs, axis=0)
e1 = axis / np.linalg.norm(axis)
allv = np.array(anim_vecs + inan_vecs)
resid = allv - np.outer(allv @ e1, e1)
from sklearn.decomposition import PCA
e2 = PCA(n_components=1, random_state=0).fit(resid).components_[0]
e2 = e2 - (e2 @ e1) * e1
e2 = e2 / np.linalg.norm(e2)

def enc(v):
    p = np.array([v @ e1, v @ e2])
    n = np.linalg.norm(p)
    return (p / n).astype(complex)

def pref_from(words):
    vs = [v for v in (vec(w) for w in words) if v is not None]
    if len(vs) < 3:
        return None
    return enc(np.mean(vs, axis=0))

CAND = {k: fs[::max(1, len(fs) // MAX_TRY_PER_VERB)][:MAX_TRY_PER_VERB]
        for k, fs in frames.items() if len(fs) >= MIN_FRAMES}

def swap_sent(sent):
    t = sent.split()
    return " ".join([t[0], t[2], t[1]])

all_sents = []
for fs in CAND.values():
    for f, _, _, _ in fs:
        all_sents += [f, swap_sent(f)]
print(f"[34a] parsing {len(all_sents)} sentences (orig+swapped) for "
      f"{len(CAND)} verbs", flush=True)
diagrams = exp13.sentences_to_diagrams(all_sents, log_interval=50)
dmap = dict(zip(all_sents, diagrams))
ansatz = exp13.make_ansatz(1, 1)

def build(sent, SUBJ):
    d = dmap.get(sent)
    if d is None:
        return None
    try:
        c = ansatz(exp13._remove_cups(d))
    except Exception:
        return None
    model = NumpyModel.from_diagrams([c], use_jit=False)
    names = [str(s) for s in model.symbols]
    if not any("s@n.l@n.l" in nm for nm in names):
        return None
    sym_index = {nm: i for i, nm in enumerate(names)}
    prog = ns["compile_circuit"](c, sym_index)
    info, _ = ns["track_wires"](prog, names)
    try:
        swid = ns["find_subject_wire"](prog, names, info, SUBJ)
    except AssertionError:
        return None
    w = ns["weights_for"](names)
    vidx = sorted(i for i, nm in enumerate(names) if "s@n.l@n.l" in nm)
    return dict(names=names, prog=prog, info=info, swid=swid, w=w, vidx=vidx)

def make_zx4(G):
    prog, w, vidx = G["prog"], G["w"], G["vidx"][:2]
    vset = set(vidx)
    ents = [oi for oi, (k, _, a) in enumerate(prog)
            if k == "CRz" and a in vset]
    assert len(ents) == 2
    n0 = len(w)
    extra = {ents[0]: n0, ents[1]: n0 + 1}
    newprog, imap = [], {}
    for oi, (k, off, a) in enumerate(prog):
        imap[oi] = len(newprog)
        newprog.append((k, off, a))
        if oi in extra:
            newprog.append(("CRx", off, extra[oi]))
    H = dict(G)
    H["prog"] = newprog
    H["w"] = np.concatenate([w, [0.0, 0.0]])
    rec = dict(G["info"][G["swid"]])
    rec["symops"] = [imap[o] for o in rec["symops"]]
    rec["ket"] = imap[rec["ket"]]
    info2 = dict(G["info"])
    info2[G["swid"]] = rec
    H["info"] = info2
    H["solve_idx"] = vidx + [n0, n0 + 1]
    return H

def run_sf(G, w, input_vec):
    H_M, FIXED1, FIXED2 = ns["H_M"], ns["FIXED1"], ns["FIXED2"]
    prog, info, swid = G["prog"], G["info"], G["swid"]
    skip = set(info[swid]["symops"])
    st = np.array(1.0 + 0j)
    for oi, (kind, off, arg) in enumerate(prog):
        if oi in skip:
            continue
        if kind == "ket":
            v_ = input_vec if oi == info[swid]["ket"] else KETS[arg]
            st = np.moveaxis(np.tensordot(st, v_, 0), -1, off)
        elif kind == "bra":
            st = np.take(st, arg, axis=off)
        elif kind == "scalar":
            st = st * arg
        elif kind == "H":
            st = apply1(st, H_M, off)
        elif kind == "fixed1":
            st = apply1(st, FIXED1[arg], off)
        elif kind == "SWAP":
            st = np.swapaxes(st, off, off + 1)
        elif kind in ("Rx", "Rz"):
            st = apply1(st, FORMS[kind](w[arg]), off)
        elif kind in ("CRz", "CRx"):
            st = apply2(st, FORMS[kind](w[arg]), off, off + 1)
        else:
            st = apply2(st, FIXED2[arg], off, off + 1)
    return np.asarray(st)

def evidence_dir(G, theta):
    w = G["w"].copy()
    for k, i in enumerate(G["solve_idx"]):
        w[i] = theta[k]
    cols = [run_sf(G, w, KETS[b]).flatten() for b in (0, 1)]
    M = np.stack(cols, axis=1)
    r = M[S_OUT, :].conj()
    n = np.linalg.norm(r)
    return (r / n if n > 1e-12 else r), n

def align(G, theta, tgt):
    d_, _ = evidence_dir(G, theta)
    return abs(np.vdot(d_, tgt)) ** 2

def fit(Gs, tgt, n_starts=6):
    dim = len(Gs[0]["solve_idx"])
    def loss(th):
        return 1 - np.mean([align(G, th, tgt) for G in Gs])
    best = None
    for s in range(n_starts):
        r = minimize(loss, np.random.default_rng(s).uniform(0, 2, dim),
                     method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12,
                              "maxiter": 8000})
        if best is None or r.fun < best.fun:
            best = r
    return best

USABLE = {}
for k, fs in CAND.items():
    pairs = []
    for sent, subj, obj, split in fs:
        Go = build(sent, subj)
        Gs_ = build(swap_sent(sent), obj)   # swapped: obj is the new subject
        if Go is not None and Gs_ is not None:
            pairs.append(dict(orig=make_zx4(Go), swap=make_zx4(Gs_),
                              sent=sent, subj=subj, obj=obj))
        if len(pairs) >= MAX_USE_PER_VERB:
            break
    print(f"[34a] {surface[k]}: usable orig+swap pairs {len(pairs)}/{len(fs)}",
          flush=True)
    if len(pairs) >= MIN_FRAMES:
        USABLE[k] = pairs

OUT = {"verbs": {}, "config": {"S_OUT": S_OUT}}
tot_q = tot_qf = tot_c = tot_n = 0
for k, pairs in USABLE.items():
    v = surface[k]
    q_correct = qf_correct = c_correct = n_pairs = 0
    rows = []
    for i, p in enumerate(pairs):
        words = [pp["subj"] for j, pp in enumerate(pairs) if j != i]
        words += [w for w in subjects[k]
                  if w not in [pp["subj"] for pp in pairs]]
        tgt = pref_from(words)
        if tgt is None:
            continue
        fb = fit([pp["orig"] for j, pp in enumerate(pairs) if j != i], tgt)
        vs_, vo_ = vec(p["subj"]), vec(p["obj"])
        if vs_ is None or vo_ is None:
            continue
        # argument-fed score: actual occupant of the subject wire goes in
        ao = align(p["orig"], fb.x, enc(vs_))
        asw = align(p["swap"], fb.x, enc(vo_))
        # filter-only control: argument-blind by construction, ~50%
        aof = align(p["orig"], fb.x, tgt)
        aswf = align(p["swap"], fb.x, tgt)
        co = abs(np.vdot(enc(vs_), tgt)) ** 2
        csw = abs(np.vdot(enc(vo_), tgt)) ** 2
        n_pairs += 1
        q_correct += int(ao > asw)
        qf_correct += int(aof > aswf)
        c_correct += int(co > csw)
        rows.append(dict(sent=p["sent"], q_orig=float(ao), q_swap=float(asw),
                         qf_orig=float(aof), qf_swap=float(aswf),
                         c_orig=float(co), c_swap=float(csw)))
    OUT["verbs"][v] = {"n": int(n_pairs), "quantum_argfed": int(q_correct),
                       "quantum_filter_control": int(qf_correct),
                       "classical_struct": int(c_correct), "pairs": rows}
    tot_q += q_correct; tot_qf += qf_correct
    tot_c += c_correct; tot_n += n_pairs
    print(f"[34a] {v}: n={n_pairs} | zx4 arg-fed {q_correct}/{n_pairs} | "
          f"filter-control {qf_correct}/{n_pairs} (expect ~chance) | "
          f"structured-classical {c_correct}/{n_pairs} | "
          f"bag-of-vectors forced 50%", flush=True)

def binom_p(k_, n_):
    return sum(math.comb(n_, j) for j in range(k_, n_ + 1)) / 2 ** n_

OUT["pooled"] = {"n": int(tot_n), "quantum_argfed": int(tot_q),
                 "quantum_filter_control": int(tot_qf),
                 "classical_struct": int(tot_c),
                 "p_quantum": binom_p(tot_q, tot_n) if tot_n else None,
                 "p_filter": binom_p(tot_qf, tot_n) if tot_n else None,
                 "p_classical": binom_p(tot_c, tot_n) if tot_n else None}
print(f"[34a] POOLED: zx4 arg-fed {tot_q}/{tot_n} "
      f"(binom p={binom_p(tot_q, tot_n):.2e}) | filter-control "
      f"{tot_qf}/{tot_n} (p={binom_p(tot_qf, tot_n):.2e}, argument-blind "
      f"by construction) | structured-classical {tot_c}/{tot_n} "
      f"(p={binom_p(tot_c, tot_n):.2e}) | bag-of-vectors 50% by "
      f"construction", flush=True)
json.dump(OUT, open("results_exp34a.json", "w"), indent=2, ensure_ascii=False)
print("[34a] DONE", flush=True)
