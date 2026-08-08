# -*- coding: utf-8 -*-
"""Exp 30b: joint multi-frame verb-parameter solve.

exp30 found: per-frame solves reach corpus preference directions
(residual -> 0) but single-frame-solved params did not keep their direction
in a new frame (n=2, mean alignment 0.50). exp30b is the decisive,
overdetermined version:

  * frame supply scaled: all label-identified VSO transitive sentences per
    verb (WordOrderMatched/WordOrder _VSO + WSD/WSD_v2 verb-initial), built
    through the real exp28a compile/surgery machinery;
  * supervised animacy axis (attested-subject centroid minus attested-object
    centroid) replaces unsupervised PCA-2 as the 2D encoding basis;
  * T1b: per-frame reachability (individual solves, residual per frame);
  * T2b: animacy relation of sense surrogates in the supervised basis;
  * T3b: JOINT solve — one theta per verb fit on ALL its frames at once.
    2 params vs M>=4 frames is overdetermined: a low joint residual cannot
    be memorization; it means one parameter pair points the evidence-row at
    the corpus preference in every frame simultaneously;
  * T4b: leakage-free leave-one-out — theta fit on M-1 frames with the
    preference vector rebuilt WITHOUT the held-out sentence's subject;
    alignment measured on the held-out frame;
  * controls: random-theta null (same reachable family, U(0,2)^2) and
    cross-verb mismatched-theta control (verb v's joint theta evaluated on
    verb w's frames against pref_w).
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

S_OUT = int(os.environ.get("S_OUT", "0"))
FUSION = os.environ.get("ARABIC_POS_FUSION", "0") == "1"
TAG = "_fusion" if FUSION else ""
MAX_TRY_PER_VERB = 24      # frames attempted through the parser per verb
MAX_USE_PER_VERB = 10      # usable frames kept for the solves
MIN_FRAMES = 4             # verbs below this are reported but not solved
N_NULL = 300               # random-theta draws for the null

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

# ── harvest: label-identified frames + subjects + objects ────────────────
frames = defaultdict(list)     # hn(verb) -> [(sentence, subj, split)]
subjects = defaultdict(list)   # hn(verb) -> [subj token]
objects_pool = []              # object head tokens (inanimate pool)
surface = {}                   # hn(verb) -> display surface

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
            # WSD splits are SVO (subj verb obj); reorder to the equally
            # grammatical VSO surface so the s@n.l@n.l gate machinery applies
            k = hn(t[1])
            surface.setdefault(k, t[1])
            vso = " ".join([t[1], t[0]] + t[2:])
            frames[k].append((vso, t[0], split))
            subjects[k].append(t[0])
            objects_pool.append(t[2])
        elif lab.endswith("_SVO"):
            subjects[hn(t[1])].append(t[0])

print("[30b] frame supply:",
      {surface[k]: len(v) for k, v in
       sorted(frames.items(), key=lambda kv: -len(kv[1]))}, flush=True)
print("[30b] subject counts:",
      {surface.get(k, k): len(v) for k, v in subjects.items()
       if k in frames}, flush=True)

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

# ── supervised 2D basis: animacy axis + dominant residual direction ──────
anim_words = sorted({w for k in frames for w in subjects[k]})
inan_words = sorted(set(objects_pool))
anim_vecs = [v for v in (vec(w) for w in anim_words) if v is not None]
inan_vecs = [v for v in (vec(w) for w in inan_words) if v is not None]
print(f"[30b] animacy pools: {len(anim_vecs)}/{len(anim_words)} subject "
      f"vectors, {len(inan_vecs)}/{len(inan_words)} object vectors",
      flush=True)
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

# ── T2b: animacy relation of sense surrogates in the supervised basis ────
SURR = {"man": "رجال", "leg": "رجله", "camel": "جمل", "beauty": "جمال"}
sense = {k: enc(vec(w)) for k, w in SURR.items() if vec(w) is not None}
T2B = {}
for k in sorted(frames, key=lambda k: -len(frames[k])):
    p = pref_from(subjects[k])
    if p is None:
        continue
    sm = abs(np.vdot(p, sense["man"])) ** 2
    sl = abs(np.vdot(p, sense["leg"])) ** 2
    sc = abs(np.vdot(p, sense["camel"])) ** 2
    sb = abs(np.vdot(p, sense["beauty"])) ** 2
    T2B[surface[k]] = {"man": sm, "leg": sl, "camel": sc, "beauty": sb,
                       "man>leg": bool(sm > sl),
                       "camel>beauty": bool(sc > sb)}
    print(f"[30b] T2b {surface[k]}: man={sm:.3f} leg={sl:.3f} "
          f"(man>leg: {sm > sl}) | camel={sc:.3f} beauty={sb:.3f} "
          f"(camel>beauty: {sc > sb})", flush=True)

# ── circuit build machinery (as exp30) ───────────────────────────────────
CAND = {k: fs[::max(1, len(fs) // MAX_TRY_PER_VERB)][:MAX_TRY_PER_VERB]
        for k, fs in frames.items() if len(fs) >= MIN_FRAMES}
all_sents = [f for fs in CAND.values() for f, _, _ in fs]
print(f"[30b] parsing {len(all_sents)} candidate frames for "
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
    vidx = [i for i, nm in enumerate(names) if "s@n.l@n.l" in nm]
    return dict(c=c, names=names, sym_index=sym_index, prog=prog,
                info=info, swid=swid, w=w, vidx=sorted(vidx))

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
    for k, i in enumerate(G["vidx"][:2]):
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
    def loss(th):
        return 1 - np.mean([align(G, th, tgt) for G in Gs])
    best = None
    for s in range(n_starts):
        r = minimize(loss, np.random.default_rng(s).uniform(0, 2, 2),
                     method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12,
                              "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    return best

# ── build usable frames per verb ─────────────────────────────────────────
USABLE = {}
for k, fs in CAND.items():
    gs = []
    for sent, subj, split in fs:
        G = build(sent, subj)
        if G is not None:
            gs.append(dict(G=G, sent=sent, subj=subj, split=split))
        if len(gs) >= MAX_USE_PER_VERB:
            break
    print(f"[30b] {surface[k]}: usable frames {len(gs)}/{len(fs)} attempted",
          flush=True)
    if len(gs) >= MIN_FRAMES:
        USABLE[k] = gs

OUT = {"T2b": T2B, "verbs": {}, "cross_verb": {},
       "config": {"S_OUT": S_OUT, "max_use": MAX_USE_PER_VERB,
                  "min_frames": MIN_FRAMES, "n_null": N_NULL}}

# ── T1b/T3b/T4b per verb ─────────────────────────────────────────────────
rng_null = np.random.default_rng(123)
NULL_THETAS = rng_null.uniform(0, 2, (N_NULL, 2))
joint_theta = {}
for k, gs in USABLE.items():
    v = surface[k]
    tgt = pref_from(subjects[k])
    if tgt is None:
        continue
    Gs = [g["G"] for g in gs]
    # T1b: individual reachability
    indiv = [float(fit([G], tgt).fun) for G in Gs]
    # T3b: joint solve on all frames
    jb = fit(Gs, tgt, n_starts=8)
    j_aligns = [float(align(G, jb.x, tgt)) for G in Gs]
    joint_theta[k] = jb.x
    # T4b: leakage-free LOO
    loo = []
    for i in range(len(gs)):
        words = [w for j, g in enumerate(gs) for w in ([g["subj"]] if j != i
                 else [])]
        words += [w for w in subjects[k]
                  if w not in [g["subj"] for g in gs]]
        tgt_tr = pref_from(words)
        if tgt_tr is None:
            continue
        fb = fit([G for j, G in enumerate(Gs) if j != i], tgt_tr)
        loo.append(float(align(Gs[i], fb.x, tgt_tr)))
    # null: random thetas, mean alignment over this verb's frames
    null_means = np.array([np.mean([align(G, th, tgt) for G in Gs])
                           for th in NULL_THETAS])
    obs = float(np.mean(loo)) if loo else float("nan")
    p = float((1 + np.sum(null_means >= obs)) / (1 + N_NULL))
    OUT["verbs"][v] = {
        "n_frames": len(gs),
        "frames": [dict(sent=g["sent"], subj=g["subj"], split=g["split"])
                   for g in gs],
        "T1b_indiv_residuals": indiv,
        "T3b_joint_residual": float(jb.fun),
        "T3b_theta": [float(x) for x in jb.x],
        "T3b_per_frame_alignment": j_aligns,
        "T4b_loo_alignments": loo,
        "T4b_loo_mean": obs,
        "null_mean": float(null_means.mean()),
        "null_p95": float(np.quantile(null_means, 0.95)),
        "p_value": p,
    }
    print(f"[30b] {v}: M={len(gs)} | T1b indiv residual "
          f"med={np.median(indiv):.2e} max={max(indiv):.2e} | "
          f"T3b joint residual={jb.fun:.4f} theta={jb.x.round(3)} | "
          f"T4b LOO mean={obs:.4f} (null mean={null_means.mean():.3f}, "
          f"p95={np.quantile(null_means, 0.95):.3f}, p={p:.4f})",
          flush=True)

# ── cross-verb mismatched-theta control ──────────────────────────────────
for k, gs in USABLE.items():
    v = surface[k]
    tgt = pref_from(subjects[k])
    row = {}
    for k2 in USABLE:
        if k2 == k or k2 not in joint_theta:
            continue
        row[surface[k2]] = float(np.mean(
            [align(g["G"], joint_theta[k2], tgt) for g in gs]))
    matched = OUT["verbs"][v]["T3b_per_frame_alignment"]
    OUT["cross_verb"][v] = {"matched_mean": float(np.mean(matched)),
                            "mismatched": row}
    if row:
        print(f"[30b] XV {v}: matched={np.mean(matched):.3f} vs "
              f"mismatched={np.mean(list(row.values())):.3f} "
              f"({ {a: round(b,3) for a,b in row.items()} })", flush=True)

ms = [OUT["verbs"][surface[k]]["T4b_loo_mean"] for k in USABLE
      if surface[k] in OUT["verbs"]]
if ms:
    print(f"[30b] SUMMARY: LOO transfer mean={np.mean(ms):.4f} over "
          f"{len(ms)} verbs (random-theta null ~"
          f"{np.mean([OUT['verbs'][surface[k]]['null_mean'] for k in USABLE]):.3f})",
          flush=True)
OUT["config"]["pos_fusion"] = FUSION
json.dump(OUT, open(f"results_exp30b_s{S_OUT}{TAG}.json", "w"), indent=2,
          ensure_ascii=False)
print("[30b] DONE", flush=True)
