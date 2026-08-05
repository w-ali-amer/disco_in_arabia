# -*- coding: utf-8 -*-
"""Exp 38: analog-native mixed ansatz — CRz + symmetric XX entangler.

exp37 showed global-drive Rydberg evolution is atom-exchange-symmetric and
therefore cannot exactly compile the asymmetric CRx component (قطع stalls
at F=0.79).  exp38 asks whether the exchange-SYMMETRIC non-diagonal
entangler XX(phi)=exp(-i pi phi/2 X.X) recombines specificity and
frame-invariance the way CRx did in exp33 zx4.  If yes, the encoder is
analog-native end to end.

exp30c (diagonal CRz): verb-SPECIFIC (matched>mismatched 6/6) but partially
frame-dependent (joint residual grows with M).  exp32 (CRx): perfectly
frame-INVARIANT ceiling (per-frame optimum identical across all frames) but
verb-degenerate (matched==mismatched; optimized thetas interchangeable).
exp33 tests whether a mixed gate family recombines specificity with
invariance, on the same 6 verbs / same frames / same nulls:

  zz  — pure CRz baseline (2 params).  Harness regression check: must
        reproduce exp30c numbers exactly.
  zx  — entangler 1 stays CRz, entangler 2 becomes CRx (2 params).
  xz  — entangler 1 becomes CRx, entangler 2 stays CRz (2 params).
  zx4 — each entangler becomes CRz(theta)·CRx(phi) (4 params).  CRz·CRz
        collapses to one angle, so this family's extra capacity is purely
        non-diagonal: any gain over zz is attributable to the CRx
        component, not parameter count.

Pre-registered readout: a variant RECOMBINES if verb-specificity
(mean matched-minus-mismatched) stays >= 0.05 AND mean joint residual
<= the zz baseline's.  Plateau signature (med==max of per-frame
residuals) diagnoses inherited CRx degeneracy.

Run with ARABIC_POS_FUSION=1 (frame supply requires the parser fusion).
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

S_OUT = int(os.environ.get("S_OUT", "0"))
if os.environ.get("ARABIC_POS_FUSION", "0") != "1":
    print("[38] WARNING: ARABIC_POS_FUSION is off — frame supply will be "
          "starved; exp33 is designed to run with fusion on.", flush=True)
MAX_TRY_PER_VERB = 24
MAX_USE_PER_VERB = 10
MIN_FRAMES = 4
N_NULL = 300
VARIANTS = ["zxx4"]

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

# ── harvest (identical to exp30b/c) ──────────────────────────────────────
frames = defaultdict(list)
subjects = defaultdict(list)
objects_pool = []
surface = {}
for split, ds in data_all.items():
    for d in ds:
        t = d["sentence"].split()
        lab = d.get("label", "")
        if len(t) < 3:
            continue
        if lab.endswith("_VSO"):
            k = hn(t[0])
            surface.setdefault(k, t[0])
            frames[k].append((d["sentence"], t[1], split))
            subjects[k].append(t[1])
            objects_pool.append(t[2])
        elif lab.startswith("WSD_") and hn(t[1]) == hn(lab.split("_")[1]):
            k = hn(t[1])
            surface.setdefault(k, t[1])
            vso = " ".join([t[1], t[0]] + t[2:])
            frames[k].append((vso, t[0], split))
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

# ── circuit build (identical to exp30b) ──────────────────────────────────
CAND = {k: fs[::max(1, len(fs) // MAX_TRY_PER_VERB)][:MAX_TRY_PER_VERB]
        for k, fs in frames.items() if len(fs) >= MIN_FRAMES}
all_sents = [f for fs in CAND.values() for f, _, _ in fs]
print(f"[38] parsing {len(all_sents)} candidate frames for "
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

XX_M = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])).astype(complex)
def XXG(t):
    return (np.cos(np.pi * t / 2) * np.eye(4, dtype=complex)
            - 1j * np.sin(np.pi * t / 2) * XX_M)

def make_variant(G, variant):
    """Return a per-variant copy of G with rewritten prog / solve_idx."""
    prog, w, vidx = G["prog"], G["w"], G["vidx"][:2]
    vset = set(vidx)
    ents = [oi for oi, (k, _, a) in enumerate(prog)
            if k == "CRz" and a in vset]
    assert len(ents) == 2, f"expected 2 verb entanglers, got {ents}"
    H = dict(G)
    if variant == "zz":
        H["solve_idx"] = vidx
    elif variant in ("zx", "xz"):
        pos = {ents[1] if variant == "zx" else ents[0]}
        H["prog"] = [(("CRx" if oi in pos else k), off, a)
                     for oi, (k, off, a) in enumerate(prog)]
        H["solve_idx"] = vidx
    elif variant in ("zx4", "zxx4"):
        kind_new = "CRx" if variant == "zx4" else "XX"
        n0 = len(w)
        extra = {ents[0]: n0, ents[1]: n0 + 1}
        newprog, imap = [], {}
        for oi, (k, off, a) in enumerate(prog):
            imap[oi] = len(newprog)
            newprog.append((k, off, a))
            if oi in extra:
                newprog.append((kind_new, off, extra[oi]))
        H["prog"] = newprog
        H["w"] = np.concatenate([w, [0.0, 0.0]])
        # remap the op indices recorded for the subject-wire surgery
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
        elif kind == "XX":
            st = apply2(st, XXG(w[arg]), off, off + 1)
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
                              "maxiter": 2000 * dim})
        if best is None or r.fun < best.fun:
            best = r
    return best

# ── usable frames (shared across variants) ───────────────────────────────
USABLE = {}
for k, fs in CAND.items():
    gs = []
    for sent, subj, split in fs:
        G = build(sent, subj)
        if G is not None:
            gs.append(dict(G=G, sent=sent, subj=subj, split=split))
        if len(gs) >= MAX_USE_PER_VERB:
            break
    print(f"[38] {surface[k]}: usable frames {len(gs)}/{len(fs)} attempted",
          flush=True)
    if len(gs) >= MIN_FRAMES:
        USABLE[k] = gs

OUT = {"variants": {}, "config": {"S_OUT": S_OUT, "n_null": N_NULL,
                                  "variants": VARIANTS}}
for variant in VARIANTS:
    VOUT = {"verbs": {}, "cross_verb": {}}
    joint_theta = {}
    for k, gs in USABLE.items():
        v = surface[k]
        tgt = pref_from(subjects[k])
        if tgt is None:
            continue
        Hs = [make_variant(g["G"], variant) for g in gs]
        dim = len(Hs[0]["solve_idx"])
        indiv = [float(fit([H], tgt).fun) for H in Hs]
        jb = fit(Hs, tgt, n_starts=8)
        j_aligns = [float(align(H, jb.x, tgt)) for H in Hs]
        joint_theta[k] = jb.x
        loo = []
        for i in range(len(gs)):
            words = [g["subj"] for j, g in enumerate(gs) if j != i]
            words += [w for w in subjects[k]
                      if w not in [g["subj"] for g in gs]]
            tgt_tr = pref_from(words)
            if tgt_tr is None:
                continue
            fb = fit([H for j, H in enumerate(Hs) if j != i], tgt_tr)
            loo.append(float(align(Hs[i], fb.x, tgt_tr)))
        null_th = np.random.default_rng(123).uniform(0, 2, (N_NULL, dim))
        null_means = np.array([np.mean([align(H, th, tgt) for H in Hs])
                               for th in null_th])
        obs = float(np.mean(loo)) if loo else float("nan")
        p = float((1 + np.sum(null_means >= obs)) / (1 + N_NULL))
        VOUT["verbs"][v] = {
            "n_frames": len(gs), "dim": dim,
            "T1b_indiv_residuals": indiv,
            "T3b_joint_residual": float(jb.fun),
            "T3b_theta": [float(x) for x in jb.x],
            "T3b_per_frame_alignment": j_aligns,
            "T4b_loo_mean": obs, "T4b_loo_alignments": loo,
            "null_mean": float(null_means.mean()),
            "null_p95": float(np.quantile(null_means, 0.95)),
            "p_value": p,
        }
        print(f"[38] {variant} {v}: M={len(gs)} d={dim} | indiv "
              f"med={np.median(indiv):.3f} max={max(indiv):.3f} | "
              f"joint={jb.fun:.4f} | LOO={obs:.4f} "
              f"(null={null_means.mean():.3f}, p={p:.4f})", flush=True)
    for k, gs in USABLE.items():
        v = surface[k]
        if v not in VOUT["verbs"]:
            continue
        tgt = pref_from(subjects[k])
        Hs = [make_variant(g["G"], variant) for g in gs]
        row = {}
        for k2 in USABLE:
            if k2 == k or k2 not in joint_theta:
                continue
            th2 = joint_theta[k2]
            if len(th2) != len(Hs[0]["solve_idx"]):
                continue
            row[surface[k2]] = float(np.mean(
                [align(H, th2, tgt) for H in Hs]))
        matched = VOUT["verbs"][v]["T3b_per_frame_alignment"]
        VOUT["cross_verb"][v] = {"matched_mean": float(np.mean(matched)),
                                 "mismatched": row}
    spec = [VOUT["cross_verb"][x]["matched_mean"]
            - np.mean(list(VOUT["cross_verb"][x]["mismatched"].values()))
            for x in VOUT["cross_verb"] if VOUT["cross_verb"][x]["mismatched"]]
    jres = [VOUT["verbs"][x]["T3b_joint_residual"] for x in VOUT["verbs"]]
    loom = [VOUT["verbs"][x]["T4b_loo_mean"] for x in VOUT["verbs"]]
    VOUT["summary"] = {"mean_specificity": float(np.mean(spec)) if spec else None,
                       "n_spec_positive": int(sum(s > 0 for s in spec)),
                       "mean_joint_residual": float(np.mean(jres)),
                       "mean_loo": float(np.nanmean(loom))}
    print(f"[38] {variant} SUMMARY: specificity={np.mean(spec):.4f} "
          f"({sum(s > 0 for s in spec)}/{len(spec)} verbs positive) | "
          f"mean joint residual={np.mean(jres):.4f} | "
          f"mean LOO={np.nanmean(loom):.4f}", flush=True)
    OUT["variants"][variant] = VOUT

json.dump(OUT, open(f"results_exp38_s{S_OUT}.json", "w"), indent=2,
          ensure_ascii=False)
print("[38] DONE", flush=True)
