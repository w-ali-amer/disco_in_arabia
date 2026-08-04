"""Exp 24: ambiguity as mixedness. WSD_v2 (4 verbs x 2 senses x 25) through
IQP L1 discard circuits; von Neumann entropy of each sentence density matrix.

Pre-registered (design doc 14_exp22_24_design.md):
H1 ancilla circuits (n_ancillas=1) more mixed than no-ancilla (paired Wilcoxon)
H2 qaTa3 (قطع, the sense-symmetric verb) has the highest median entropy among
   the four verbs (rank + permutation test over verb labels)
H3 within-verb entropy differs by sense class (MWU two-sided, Bonferroni x4)
Grounding: paper section 8.3 (ancilla-traced density matrices as ambiguity).
"""
import json, os, time
import numpy as np
from collections import defaultdict

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel
from scipy.stats import wilcoxon, mannwhitneyu

RNG = np.random.default_rng(42)
data = json.load(open("sentences.json"))["WordSenseDisambiguation_v2"]
sents = [d["sentence"] for d in data]
verbs = [d["verb"] for d in data]
senses = [d["sense"] for d in data]
print(f"[24] {len(sents)} WSD_v2 sentences, verbs: {sorted(set(verbs))}",
      flush=True)

diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)


def circuits_for(n_anc):
    ansatz = exp13.IQPAnsatz({exp13.S_ty: 1, exp13.N_ty: 1}, n_layers=1,
                             discard=True, n_ancillas=n_anc)
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


def to_dm(T, is_mixed):
    T = np.asarray(T)
    if not is_mixed:
        v = T.flatten()
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-12 else v
        return np.outer(v, v.conj())
    d = int(round(T.size ** 0.5))
    cands = [T.reshape(d, d)]
    if T.ndim == 4:
        cands.append(np.transpose(T, (0, 2, 1, 3)).reshape(d, d))
    for rho in cands:
        if (np.linalg.norm(rho - rho.conj().T) < 1e-8 and
                np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -1e-8):
            tr = float(np.real(np.trace(rho)))
            return rho / tr if tr > 1e-12 else rho
    raise ValueError("no valid DM ordering")


def entropy(rho):
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    ev = np.clip(np.real(ev), 1e-12, 1.0)
    ev = ev / ev.sum()
    return float(-np.sum(ev * np.log2(ev)))


OUT = {"n_sentences": len(sents)}
TL = {"رفع": "rafa3", "حمل": "Hamal", "قطع": "qaTa3", "ضرب": "Daraba"}
for mode in ("legacy", "fixed"):
    os.environ["QFM_WARMSTART"] = mode
    res = {}
    ent = {}
    valid = {}
    for n_anc in (0, 1):
        circs, vidx = circuits_for(n_anc)
        model = NumpyModel.from_diagrams(circs, use_jit=False)
        names = [str(s) for s in model.symbols]
        w = exp13.warmstart_weights(model)
        wmap = dict(zip(names, w))
        S = np.full(len(sents), np.nan)
        for k, c in enumerate(circs):
            syms = sorted(c.free_symbols, key=str)
            vals = [wmap[str(s)] for s in syms]
            T = (c.lambdify(*syms)(*vals).eval() if syms else c.eval())
            S[vidx[k]] = entropy(to_dm(T, c.is_mixed))
        ent[n_anc] = S
        valid[n_anc] = set(vidx)
        print(f"[24] {mode} anc={n_anc}: {len(vidx)} circuits, "
              f"median entropy {np.nanmedian(S):.4f}", flush=True)
    both = sorted(valid[0] & valid[1])
    d0 = ent[0][both]
    d1 = ent[1][both]
    try:
        wstat, pw = wilcoxon(d1, d0)
    except ValueError:
        pw = 1.0
    res["H1_anc_vs_noanc"] = {"median_anc": float(np.median(d1)),
                              "median_noanc": float(np.median(d0)),
                              "wilcoxon_p": float(pw), "n": len(both)}
    print(f"[24] {mode} H1: anc {np.median(d1):.4f} vs noanc "
          f"{np.median(d0):.4f} (p={pw:.2e})", flush=True)
    # H2/H3 on the ancilla family (the ambiguity-bearing construction)
    per_verb = defaultdict(list)
    per_vs = defaultdict(list)
    table = []
    for i in both:
        per_verb[verbs[i]].append(ent[1][i])
        per_vs[(verbs[i], senses[i])].append(ent[1][i])
        table.append({"sentence": sents[i], "verb": verbs[i],
                      "sense": senses[i], "S_anc": float(ent[1][i]),
                      "S_noanc": float(ent[0][i])})
    med = {v: float(np.median(x)) for v, x in per_verb.items()}
    order = sorted(med, key=med.get, reverse=True)
    obs_rank = order.index("قطع") if "قطع" in order else -1
    labels_v = [verbs[i] for i in both]
    vals_v = np.array([ent[1][i] for i in both])
    perm_top = 0
    for _ in range(10000):
        lv = RNG.permutation(labels_v)
        m = defaultdict(list)
        for l, x in zip(lv, vals_v):
            m[l].append(x)
        mm = {v: np.median(x) for v, x in m.items()}
        if max(mm, key=mm.get) == "قطع":
            perm_top += 1
    res["H2_qata3_most_mixed"] = {
        "verb_medians": {TL.get(v, v): m for v, m in med.items()},
        "qata3_rank": obs_rank + 1,
        "perm_p_top1": float((perm_top + 1) / 10001)}
    print(f"[24] {mode} H2: medians "
          f"{ {TL.get(v,v): round(m,4) for v,m in med.items()} } "
          f"qaTa3 rank {obs_rank+1}", flush=True)
    h3 = {}
    for v in sorted(set(verbs)):
        groups = [per_vs[(v, s)] for s in sorted({senses[i] for i in both
                                                  if verbs[i] == v})]
        if len(groups) == 2 and min(len(g) for g in groups) > 3:
            u, p = mannwhitneyu(groups[0], groups[1],
                                alternative="two-sided")
            h3[TL.get(v, v)] = {"p": float(p),
                                "sig_bonf4": bool(p < 0.0125)}
    res["H3_sense_diff"] = h3
    print(f"[24] {mode} H3: " +
          ", ".join(f"{k} p={x['p']:.4f}" for k, x in h3.items()), flush=True)
    res["table"] = table
    OUT[mode] = res
    json.dump(OUT, open("results_exp24.json", "w"), indent=2,
              ensure_ascii=False)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp24.json", "w"), indent=2, ensure_ascii=False)
print(f"[24] DONE in {OUT['runtime_sec']}s", flush=True)
