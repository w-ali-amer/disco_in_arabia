# -*- coding: utf-8 -*-
"""S1c: batch-solve the dial bank from s1_harvest.json.

For every harvested verb with enough real-corpus subject vectors:
  * preference vector in the supervised animacy basis (built from the
    harvested pools themselves);
  * 4 dials solved (zxx4 analog-native family, exp38) against ONE canonical
    frame — justified by the family's measured total frame invariance;
  * VALIDATION subsample: for the 12 best-harvested verbs, the full
    multi-frame LOO battery on real corpus sentences (fit on N−1 real
    frames, test the held-out one) — the exp38 proof repeated on real text.
Output: s1_dial_bank.json + summary.

Run in the lambeq env with ARABIC_POS_FUSION=1.
"""
import json, os
import numpy as np
from scipy.optimize import minimize

S_OUT = int(os.environ.get("S_OUT", "0"))
MIN_SUBJ = 15
N_VALIDATE = 12
MAX_VAL_FRAMES = 6

src = open("exp28a_real_gates.py", encoding="utf-8").read()
head = src[:src.index("# ── parse frames")]
ns = {}
exec(compile(head, "exp28a_head", "exec"), ns)
exp13, NumpyModel = ns["exp13"], ns["NumpyModel"]
KETS, FORMS = ns["KETS"], ns["FORMS"]
apply1, apply2 = ns["apply1"], ns["apply2"]

harvest = json.load(open("s1_harvest.json"))

def hn(w):
    return w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

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

# ── supervised basis from the harvested pools ────────────────────────────
subj_pool, obj_pool = [], []
for r in harvest.values():
    subj_pool += r["subjects"]
    obj_pool += r["objects"]
subj_pool = list(dict.fromkeys(subj_pool))[:4000]
obj_pool = list(dict.fromkeys(obj_pool))[:4000]
sv = [v for v in (vec(w) for w in subj_pool) if v is not None]
ov = [v for v in (vec(w) for w in obj_pool) if v is not None]
print(f"[s1c] basis pools: {len(sv)} subject vecs, {len(ov)} object vecs",
      flush=True)
axis = np.mean(sv, axis=0) - np.mean(ov, axis=0)
e1 = axis / np.linalg.norm(axis)
allv = np.array(sv[:2000] + ov[:2000])
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
    if len(vs) < MIN_SUBJ:
        return None, len(vs)
    return enc(np.mean(vs, axis=0)), len(vs)

# ── circuit machinery (exp38 zxx4) ───────────────────────────────────────
ansatz = exp13.make_ansatz(1, 1)
XX_M = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])).astype(complex)
def XXG(t):
    return (np.cos(np.pi * t / 2) * np.eye(4, dtype=complex)
            - 1j * np.sin(np.pi * t / 2) * XX_M)

def build(sent, SUBJ, dmap):
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
    vset = set(vidx[:2])
    ents = [oi for oi, (k, _, a) in enumerate(prog)
            if k == "CRz" and a in vset]
    if len(ents) != 2:
        return None
    n0 = len(w)
    extra = {ents[0]: n0, ents[1]: n0 + 1}
    newprog, imap = [], {}
    for oi, (k, off, a) in enumerate(prog):
        imap[oi] = len(newprog)
        newprog.append((k, off, a))
        if oi in extra:
            newprog.append(("XX", off, extra[oi]))
    rec = dict(info[swid])
    rec["symops"] = [imap[o] for o in rec["symops"]]
    rec["ket"] = imap[rec["ket"]]
    info2 = dict(info)
    info2[swid] = rec
    return dict(prog=newprog, info=info2, swid=swid,
                w=np.concatenate([w, [0.0, 0.0]]),
                solve_idx=vidx[:2] + [n0, n0 + 1])

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

def align(G, theta, tgt):
    w = G["w"].copy()
    for k, i in enumerate(G["solve_idx"]):
        w[i] = theta[k]
    cols = [run_sf(G, w, KETS[b]).flatten() for b in (0, 1)]
    r = np.stack(cols, axis=1)[S_OUT, :].conj()
    n = np.linalg.norm(r)
    r = r / n if n > 1e-12 else r
    return abs(np.vdot(r, tgt)) ** 2

def fit(Gs, tgt, n_starts=6):
    def loss(th):
        return 1 - np.mean([align(G, th, tgt) for G in Gs])
    best = None
    for s in range(n_starts):
        r = minimize(loss, np.random.default_rng(s).uniform(0, 2, 4),
                     method="Nelder-Mead",
                     options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-11})
        if best is None or r.fun < best.fun:
            best = r
    return best

# ── canonical frames: parse once, in bulk ────────────────────────────────
verbs = [v for v, r in harvest.items() if r["n_hit"] >= MIN_SUBJ]
print(f"[s1c] {len(verbs)} verbs pass the {MIN_SUBJ}-subject bar", flush=True)
canon = {v: f"{v} الرجل الكتاب" for v in verbs}
best_harvested = sorted(verbs, key=lambda v: -harvest[v]["n_hit"])[:N_VALIDATE]
# validation frames: VSO rebuilds from real harvested (subject, object) pairs
val_sents = {v: [] for v in best_harvested}
for v in best_harvested:
    r = harvest[v]
    for subj, obj in zip(r["subjects"][:MAX_VAL_FRAMES * 2],
                         r["objects"][:MAX_VAL_FRAMES * 2]):
        val_sents[v].append((f"{v} {subj} {obj}", subj))

all_sents = list(canon.values()) + [s for v in best_harvested
                                    for s, _ in val_sents[v]]
print(f"[s1c] parsing {len(all_sents)} frames", flush=True)
diagrams = exp13.sentences_to_diagrams(all_sents, log_interval=100)
dmap = dict(zip(all_sents, diagrams))

bank, n_solved, resids = {}, 0, []
for v in verbs:
    tgt, n_vec = pref_from(harvest[v]["subjects"])
    if tgt is None:
        continue
    G = build(canon[v], "الرجل", dmap)
    if G is None:
        continue
    fb = fit([G], tgt)
    bank[v] = {"dials": [float(x) for x in fb.x],
               "residual": float(fb.fun),
               "n_subject_vectors": n_vec,
               "pref": [float(np.real(tgt[0])), float(np.real(tgt[1]))]}
    n_solved += 1
    resids.append(fb.fun)
    if n_solved % 25 == 0:
        print(f"[s1c] solved {n_solved} (median residual "
              f"{np.median(resids):.2e})", flush=True)
print(f"[s1c] BANK: {n_solved} verbs solved | median residual "
      f"{np.median(resids):.2e} | max {max(resids):.2e}", flush=True)

# ── validation battery on real corpus frames ─────────────────────────────
VAL = {}
for v in best_harvested:
    if v not in bank:
        continue
    Gs, subs = [], []
    for sent, subj in val_sents[v]:
        G = build(sent, subj, dmap)
        if G is not None:
            Gs.append(G)
            subs.append(subj)
        if len(Gs) >= MAX_VAL_FRAMES:
            break
    if len(Gs) < 4:
        print(f"[s1c] VAL {v}: only {len(Gs)} usable real frames — skipped",
              flush=True)
        continue
    loo = []
    for i in range(len(Gs)):
        tgt_tr, _ = pref_from([s for j, s in enumerate(harvest[v]["subjects"])
                               if s != subs[i]])
        if tgt_tr is None:
            continue
        fb = fit([G for j, G in enumerate(Gs) if j != i], tgt_tr)
        loo.append(float(align(Gs[i], fb.x, tgt_tr)))
    VAL[v] = {"n_frames": len(Gs), "loo_mean": float(np.mean(loo)),
              "loo": loo}
    print(f"[s1c] VAL {v}: M={len(Gs)} real frames | LOO mean "
          f"{np.mean(loo):.4f}", flush=True)
if VAL:
    print(f"[s1c] VALIDATION: mean LOO {np.mean([d['loo_mean'] for d in VAL.values()]):.4f} "
          f"over {len(VAL)} verbs on real corpus frames", flush=True)
json.dump({"bank": bank, "validation": VAL,
           "basis": {"e1_note": "animacy axis from harvested pools"}},
          open("s1_dial_bank.json", "w"), indent=1, ensure_ascii=False)
print("[s1c] DONE", flush=True)
