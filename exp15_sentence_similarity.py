"""Experiment 15 (Rung 2): Sentence meaning as geometry.
Fidelity structure of Arabic sentence states over WordOrderMatched.
Per spec 04_rung2_experiment_spec.md. Pre-registered: L0 all fidelities == 1.0 exactly.
"""
import json, time, itertools
import numpy as np

t0 = time.time()
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel

RNG = np.random.default_rng(42)
OUT = {}

data = json.load(open("sentences.json"))["WordOrderMatched"]
sents  = [d["sentence"] for d in data]
labels = [d["label"] for d in data]
pids   = [d["pair_id"] for d in data]
print(f"[exp15] {len(sents)} sentences loaded")

# --- parse once, ansatz per layer ---
diagrams = exp13.sentences_to_diagrams(sents, log_interval=999)

def circuits_for(n_layers):
    ansatz = exp13.make_ansatz(n_layers, 1)
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

def extract_states(circs):
    model = NumpyModel.from_diagrams(circs, use_jit=False)
    model.weights = exp13.warmstart_weights(model)
    wmap = dict(zip([str(s) for s in model.symbols], model.weights))
    states, norms = [], []
    for c in circs:
        syms = sorted(c.free_symbols, key=str)
        vals = [wmap[str(s)] for s in syms]
        amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
        p = float(np.sum(np.abs(amps) ** 2))
        norms.append(p)
        states.append(amps / np.sqrt(p) if p > 1e-12 else amps * np.nan)
    return np.array(states), np.array(norms), model

results = {}
for L in (0, 1):
    circs, vidx = circuits_for(L)
    states, norms, model = extract_states(circs)
    excluded = [i for i, p in zip(vidx, norms) if p < 1e-6]
    print(f"[exp15] L{L}: {len(circs)} circuits, norms median={np.median(norms):.4f} min={norms.min():.2e}, excluded={excluded}")
    results[L] = dict(states=states, norms=norms, vidx=vidx, excluded=excluded)
    np.savez(f"states_L{L}.npz", states=states, norms=norms, vidx=vidx,
             labels=np.array([labels[i] for i in vidx]),
             pair_ids=np.array([pids[i] for i in vidx]),
             sentences=np.array([sents[i] for i in vidx]))

# --- preflight symbol audit (pair 0) ---
c1, v1 = circuits_for(1)
i_svo = next(k for k, i in enumerate(v1) if pids[i] == 0 and labels[i].endswith("SVO"))
i_vso = next(k for k, i in enumerate(v1) if pids[i] == 0 and labels[i].endswith("VSO"))
s_svo = set(map(str, c1[i_svo].free_symbols)); s_vso = set(map(str, c1[i_vso].free_symbols))
audit = dict(shared=sorted(s_svo & s_vso), only_svo=sorted(s_svo - s_vso), only_vso=sorted(s_vso - s_svo))
OUT["symbol_audit_pair0"] = audit
print(f"[exp15] symbol audit pair0: shared={len(audit['shared'])} only_svo={len(audit['only_svo'])} only_vso={len(audit['only_vso'])}")

def fid(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)

# --- L0 structural control ---
S0 = results[0]["states"]
f00 = [fid(S0[i], S0[j]) for i, j in itertools.combinations(range(len(S0)), 2)]
OUT["L0_control"] = dict(min_fid=float(np.min(f00)), max_fid=float(np.max(f00)), n_pairs=len(f00))
print(f"[exp15] L0 CONTROL: fidelities min={np.min(f00):.12f} max={np.max(f00):.12f} (predicted exactly 1.0)")

# --- conditions on L1 ---
S1, v1idx = results[1]["states"], results[1]["vidx"]
lab1 = [labels[i] for i in v1idx]; pid1 = [pids[i] for i in v1idx]
by_pid = {}
for k, (p, l) in enumerate(zip(pid1, lab1)):
    by_pid.setdefault(p, {})[l.split("_")[-1]] = k

A = [(d["SVO"], d["VSO"]) for p, d in sorted(by_pid.items()) if "SVO" in d and "VSO" in d]
svo_k = [k for k, l in enumerate(lab1) if l.endswith("SVO")]
vso_k = [k for k, l in enumerate(lab1) if l.endswith("VSO")]

def sample_pairs(pool, n, forbid_same_pid=True):
    out, seen = [], set()
    while len(out) < n:
        i, j = RNG.choice(pool, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key in seen or (forbid_same_pid and pid1[i] == pid1[j]):
            continue
        seen.add(key); out.append((int(i), int(j)))
    return out

B = sample_pairs(svo_k, 30) + sample_pairs(vso_k, 30)
C = []
seen = set()
while len(C) < 60:
    i = int(RNG.choice(svo_k)); j = int(RNG.choice(vso_k))
    key = (min(i, j), max(i, j))
    if key in seen or pid1[i] == pid1[j]:
        continue
    seen.add(key); C.append((i, j))

fA = np.array([fid(S1[i], S1[j]) for i, j in A])
fB = np.array([fid(S1[i], S1[j]) for i, j in B])
fC = np.array([fid(S1[i], S1[j]) for i, j in C])
selff = [fid(S1[i], S1[i]) for i in range(len(S1))]
assert max(abs(1 - np.array(selff))) < 1e-9, "self-fidelity check failed"

def summ(x):
    return dict(median=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                mean=float(np.mean(x)), n=len(x))
OUT["fidelity"] = dict(A_matched=summ(fA), B_same_structure=summ(fB), C_random=summ(fC))
print(f"[exp15] fidelity medians: A(matched)={np.median(fA):.4f} B(same-struct)={np.median(fB):.4f} C(random)={np.median(fC):.4f}")

# --- AUC + stats ---
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

def auc_block(pos, neg, tag):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    obs = roc_auc_score(y, s)
    perm = np.empty(10000)
    for t in range(10000):
        yp = RNG.permutation(y); perm[t] = roc_auc_score(yp, s)
    pval = float((np.sum(perm >= obs) + 1) / 10001)
    boots = np.empty(10000)
    for t in range(10000):
        bi = RNG.integers(0, len(s), len(s))
        try: boots[t] = roc_auc_score(y[bi], s[bi])
        except ValueError: boots[t] = np.nan
    ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    U, p_u = mannwhitneyu(pos, neg, alternative="greater")
    r = dict(auc=float(obs), perm_p=pval, ci95=ci, mannwhitney_p=float(p_u))
    print(f"[exp15] AUC {tag}: {obs:.4f} CI{ci} perm_p={pval:.5f} MWU_p={p_u:.2e}")
    return r

OUT["auc_A_vs_B"] = auc_block(fA, fB, "A-vs-B")
OUT["auc_A_vs_C"] = auc_block(fA, fC, "A-vs-C")

# --- retrieval ---
n = len(S1)
F = np.array([[fid(S1[i], S1[j]) for j in range(n)] for i in range(n)])
top1 = 0; rr = []
partner = {}
for p, d in by_pid.items():
    if "SVO" in d and "VSO" in d:
        partner[d["SVO"]] = d["VSO"]; partner[d["VSO"]] = d["SVO"]
for i in range(n):
    if i not in partner: continue
    order = np.argsort(-np.delete(F[i], i))
    others = np.delete(np.arange(n), i)
    rank = int(np.where(others[order] == partner[i])[0][0]) + 1
    rr.append(1.0 / rank); top1 += int(rank == 1)
OUT["retrieval"] = dict(top1_acc=top1 / len(rr), mrr=float(np.mean(rr)), n=len(rr))
print(f"[exp15] retrieval: top1={OUT['retrieval']['top1_acc']:.4f} MRR={OUT['retrieval']['mrr']:.4f}")

# --- baselines ---
try:
    av = np.array([exp13.sentence_to_aravec(s) for s in [sents[i] for i in v1idx]])
    def cos(i, j):
        a, b = av[i], av[j]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else 0.0
    cA = [cos(i, j) for i, j in A]; cB = [cos(i, j) for i, j in B]; cC = [cos(i, j) for i, j in C]
    OUT["aravec"] = dict(A=summ(np.array(cA)), B=summ(np.array(cB)), C=summ(np.array(cC)),
                         A_equal_1_frac=float(np.mean(np.isclose(cA, 1.0, atol=1e-9))))
    print(f"[exp15] AraVec cosine medians: A={np.median(cA):.4f} B={np.median(cB):.4f} C={np.median(cC):.4f} (A==1.0 frac {OUT['aravec']['A_equal_1_frac']:.2f})")
except Exception as e:
    OUT["aravec"] = f"failed: {e}"
try:
    emb = exp13.get_arabert_cls([sents[i] for i in v1idx])
    def cosb(i, j):
        a, b = emb[i], emb[j]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    bA = [cosb(i, j) for i, j in A]; bB = [cosb(i, j) for i, j in B]; bC = [cosb(i, j) for i, j in C]
    OUT["arabert_frozen"] = dict(A=summ(np.array(bA)), B=summ(np.array(bB)), C=summ(np.array(bC)))
    print(f"[exp15] AraBERT cosine medians: A={np.median(bA):.4f} B={np.median(bB):.4f} C={np.median(bC):.4f}")
except Exception as e:
    OUT["arabert_frozen"] = f"failed: {e}"

OUT["frontier_reference_order_acc"] = dict(QFM_L1=0.649, AraVec=0.128, AraBERT_frozen=0.861, AraBERT_finetuned=1.0, topology_only_sym=0.642)
auc = OUT["auc_A_vs_B"]["auc"]
OUT["verdict"] = ("H1 strong" if auc >= 0.75 else "H1 weak" if auc >= 0.6 else "H1 not supported (see L1-tied variant, spec section 8)")
OUT["runtime_sec"] = round(time.time() - t0, 1)

json.dump(OUT, open("results_exp15.json", "w"), indent=2, ensure_ascii=False)

# --- figure ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([fA, fB, fC], showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["A: matched pair\n(same words, same meaning,\ndifferent order)",
                        "B: same structure,\ndifferent words", "C: random"])
    ax.set_ylabel("Quantum state fidelity |<psi_i|psi_j>|^2")
    ax.set_title(f"Arabic sentence similarity in quantum state space (QFM L1)\nAUC A-vs-B = {auc:.3f}   |   L0 control: all fidelities = 1.0 exactly")
    plt.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig4_similarity.png", dpi=200)
    print("[exp15] figure saved: figures/fig4_similarity.png")
except Exception as e:
    print(f"[exp15] figure failed: {e}")

print(f"[exp15] DONE in {OUT['runtime_sec']}s — verdict: {OUT['verdict']}")
open("exp15.DONE", "w").write(OUT["verdict"])
