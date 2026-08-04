# -*- coding: utf-8 -*-
"""Exp 30: corpus-statistics verb encoder (G&S relational recipe + exp19-
style closed-form solve) + cross-frame transfer test.

Recipe: verb preference vector = normalized mean embedding of its ATTESTED
SUBJECTS in the corpus (sentences.json, all splits), projected to the PCA-2
basis; sense states = embedding-encoded via sense-form surrogates; verb's 2
circuit params SOLVED so the gate evidence-row aligns with the preference
vector. Zero training. Tests: T1 solve residual (reachability), T2 animacy
relation survives 2D compression, T3 TRANSFER: params solved on frame 1
keep their direction in a different frame with the same verb.
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

S_OUT = int(os.environ.get("S_OUT", "0"))
src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS = ns["KETS"], ns["FORMS"]
apply1, apply2 = ns["apply1"], ns["apply2"]

data_all = json.load(open("sentences.json", encoding="utf-8"))
roots = json.load(open("exp23_roots.json"))["words"]

# harvest verb -> subject list (VSO: tok0=verb tok1=subj; SVO: tok1=verb)
VERBS = ["قرأ", "حمل", "فتح", "كتب"]
def hnorm(w):
    return (w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا"))
VN = {hnorm(v): v for v in VERBS}
subj = defaultdict(list)
for split in data_all.values():
    for d in split:
        t = d["sentence"].split()
        if len(t) >= 3:
            if hnorm(t[0]) in VN:
                subj[VN[hnorm(t[0])]].append(t[1])
            elif hnorm(t[1]) in VN:
                subj[VN[hnorm(t[1])]].append(t[0])
print("[30] attested subjects:", {v: len(s) for v, s in subj.items()},
      flush=True)

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

# PCA-2 basis over all available noun vectors (subjects + surrogates)
SURR = {"man": "رجال", "leg": "رجله", "camel": "جمل", "beauty": "جمال"}
pool_words = sorted({w for s in subj.values() for w in s} | set(SURR.values()))
pool = [(w, vec(w)) for w in pool_words]
pool = [(w, v) for w, v in pool if v is not None]
print(f"[30] vectors: {len(pool)}/{len(pool_words)} "
      f"(missing: {[w for w in pool_words if vec(w) is None][:6]})", flush=True)
from sklearn.decomposition import PCA
mat = np.array([v for _, v in pool])
pca = PCA(n_components=2, random_state=0).fit(mat)
def enc(v):
    p = pca.transform(v.reshape(1, -1))[0]
    return (p / np.linalg.norm(p)).astype(complex)

pref = {}
for v in VERBS:
    vs = [vec(w) for w in subj[v]]
    vs = [x for x in vs if x is not None]
    if len(vs) < 3:
        print(f"[30] {v}: only {len(vs)} attested-subject vectors — skipped",
              flush=True)
        continue
    pref[v] = enc(np.mean(vs, axis=0))
VERBS = [v for v in VERBS if v in pref]
sense = {k: enc(vec(w)) for k, w in SURR.items() if vec(w) is not None}
print(f"[30] senses encoded: {list(sense)}", flush=True)
# T2: does 2D compression keep the animacy relation?
for v in VERBS:
    sm = abs(np.vdot(pref[v], sense["man"])) ** 2
    sl = abs(np.vdot(pref[v], sense["leg"])) ** 2
    sc = abs(np.vdot(pref[v], sense["camel"])) ** 2
    sb = abs(np.vdot(pref[v], sense["beauty"])) ** 2
    print(f"[30] T2 {v}: man={sm:.3f} leg={sl:.3f} (man>leg: {sm>sl}) | "
          f"camel={sc:.3f} beauty={sb:.3f} (camel>beauty: {sc>sb})",
          flush=True)

FRAMES = {"قرأ": [("قرأ الطالب كتاب النحو", "الطالب"),
                  ("قرأ الولد كتاب العلوم", "الولد")],
          "حمل": [("حمل الولد حقيبة المدرسة", "الولد"),
                  ("حمل المعلم كتاب النحو", "المعلم")],
          "فتح": [("فتح المدير باب المكتب", "المدير"),
                  ("فتح الطالب باب المدرسة", "الطالب")],
          "كتب": [("كتب الطالب الدرس الجديد", "الطالب"),
                  ("كتب المدير درس اللغة", "المدير")]}
all_sents = [f for v in VERBS for f, _ in FRAMES[v]]
diagrams = exp13.sentences_to_diagrams(all_sents, log_interval=999)
dmap = dict(zip(all_sents, diagrams))
ansatz = exp13.make_ansatz(1, 1)

def build(sent, SUBJ):
    d = dmap[sent]
    if d is None:
        return None
    try:
        c = ansatz(exp13._remove_cups(d))
    except Exception:
        return None
    model = NumpyModel.from_diagrams([c], use_jit=False)
    names = [str(s) for s in model.symbols]
    if not any("s@n.l@n.l" in nm for nm in names):
        return None   # need true 3-leg verb (object present)
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

OUT = {"T2": "printed", "solves": {}, "transfer": {}}
for v in VERBS:
    G1 = build(*FRAMES[v][0])
    if G1 is None:
        print(f"[30] {v}: frame1 unusable — skip", flush=True)
        continue
    tgt = pref[v]
    def loss(th):
        d_, n_ = evidence_dir(G1, th)
        return 1 - abs(np.vdot(d_, tgt)) ** 2
    best = None
    for s in range(6):
        r = minimize(loss, np.random.default_rng(s).uniform(0, 2, 2),
                     method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    OUT["solves"][v] = {"residual": float(best.fun),
                        "theta": [float(x) for x in best.x]}
    print(f"[30] T1 {v}: solve residual={best.fun:.2e} theta={best.x.round(3)}",
          flush=True)
    G2 = build(*FRAMES[v][1])
    if G2 is None:
        print(f"[30] T3 {v}: alt frame unusable — transfer skipped", flush=True)
        continue
    d2, n2 = evidence_dir(G2, best.x)
    a = abs(np.vdot(d2, tgt)) ** 2
    OUT["transfer"][v] = {"alignment_alt_frame": float(a)}
    print(f"[30] T3 {v}: transferred direction alignment in NEW frame = "
          f"{a:.4f} (1.0 = perfect transfer, 0.5 = random)", flush=True)

good = [t["alignment_alt_frame"] for t in OUT["transfer"].values()]
if good:
    print(f"[30] TRANSFER SUMMARY: mean={np.mean(good):.4f} over "
          f"{len(good)} verbs (random baseline 0.5)", flush=True)
json.dump(OUT, open(f"results_exp30_s{S_OUT}.json", "w"), indent=2,
          ensure_ascii=False)
print("[30] DONE", flush=True)
