"""Exp 16 Phase 0: grammatical structure discrimination via atom geometry.

Design (locked 2026-08-02 after device probing):
- Device: pulser AnalogDevice, the public model of Pasqal's hardware family.
  KEY CONSTRAINT DISCOVERED: AnalogDevice exposes NO DMM channels in pulser
  1.8.0 — per-atom detuning is not available on the public device model. All
  word information therefore enters through GEOMETRY: each word displaces its
  atom perpendicular to the chain by y = ETA * a * v(word). The sentence is
  literally a shape. This is fully device-legal under one global pulse and
  makes the no-interaction null exact even for site-resolved statistics
  (identical atoms + global drive + no interaction => identical states).
- Registers: 3 atoms on a line in utterance order, spacing a (Encoding A).
  SVO twin vs VSO twin = same three displacements, different arrangement =>
  different interaction matrix (Rydberg ~ 1/r^6).
- One fixed global pulse for every register in the study (the pulse is the
  measurement apparatus; the geometry is the sample):
  T=4000 ns, Omega ramp-plateau-ramp to 2pi rad/us, detuning -4pi -> +4pi.
- Conditions:
  C0_uniform  : y=0 everywhere. SVO/VSO registers identical BY CONSTRUCTION
                => exact null at all spacings (control).
  C1_wordclass: verbs displaced by ETA*a, nouns 0. Word-order pattern only
                (all SVO registers identical: N-V-N; all VSO: V-N-N).
                "WITHOUT embeddings" condition.
  C2_embedding: y = ETA*a * PCA-1(AraVec vector of the word), scaled to [0,1].
                Per-word content displacement. "WITH embeddings" condition.
- Primary features: permutation-invariant statistics of the EXACT output
  distribution — P(total excitations n=0..3), sorted single-site marginals,
  sorted connected two-point correlators. TVD on the symmetrized P(n).
  Site-resolved TVD reported as labeled secondary.
- Pre-registered predictions:
  P1: at a=25um (V_nn/Omega ~ 5e-4) TVD -> 0 and AUC -> 0.5.
  P2: TVD and AUC rise as a decreases into the interacting regime.
- Stats: per-spacing classifier AUC (SVC-RBF, StratifiedGroupKFold by pair,
  5 folds x 10 seeds); per-pair TVD medians with IQR; all saved to JSON.

Basis note (verified by probe2.py): pulser exact final-state indexing uses the
inverted bit convention relative to sample_final_state bitstrings ('1' =
rydberg in samples). Measurement prob of bitstring b = |amp[complement(b)]|^2.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

QUICK = "--quick" in sys.argv

N_SHOTS = 1000                # per register, per the spec protocol. Classifier
                              # features use shot-sampled empirical
                              # distributions: with exact noiseless features an
                              # SVC separates 1e-6-level deterministic
                              # differences after standardization and the
                              # far-spacing null breaks (verified in --quick).
                              # TVD curves stay exact (they are honest
                              # distances with no classifier in the loop).
SHOT_RNG = np.random.default_rng(1234)

T = 4000                      # ns
OMEGA_MAX = 2 * np.pi * 1.0   # rad/us (device max is 2pi*2)
DET_RANGE = 2 * np.pi * 2.0   # rad/us global detuning ramp endpoint
ETA = 0.35                    # displacement fraction of spacing
SPACINGS = (list(np.geomspace(5.0, 25.0, 2)) if QUICK else
            sorted(set(np.round(np.geomspace(5.0, 25.0, 8), 3)) |
                   {6.8, 7.3, 7.9, 8.6, 9.3}))  # extra points resolve the
                                                # crossover peak found on the
                                                # first 8-point pass
N_SEEDS = 2 if QUICK else 10
N_PAIRS_QUICK = 5
C6 = AnalogDevice.interaction_coeff

t00 = time.time()

# ── dataset and twins ────────────────────────────────────────────────────────
data = json.load(open("sentences.json", encoding="utf-8"))["WordOrderMatched"]
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
if QUICK:
    twins = twins[:N_PAIRS_QUICK]
print(f"[16] twins: {len(twins)}", flush=True)

# ── word values ──────────────────────────────────────────────────────────────
wv = json.load(open("exp16_wordvecs.json"))["vectors"]
vocab = sorted(wv)
mat = np.array([wv[w] for w in vocab])
pca = PCA(n_components=1, random_state=0)
scores = pca.fit_transform(mat)[:, 0]
s01 = (scores - scores.min()) / (scores.max() - scores.min())
pca01 = dict(zip(vocab, (float(x) for x in s01)))
evr = float(pca.explained_variance_ratio_[0])
print(f"[16] PCA-1 explained variance: {evr:.3f}", flush=True)

verbs, nouns = set(), set()
for d in data:
    toks = d["sentence"].split()
    if d["label"].endswith("VSO"):
        verbs.add(toks[0]); nouns.update(toks[1:])
    else:
        verbs.add(toks[1]); nouns.update([toks[0], toks[2]])
overlap = verbs & nouns
print(f"[16] classes: {len(verbs)} verbs, {len(nouns)} nouns, overlap={sorted(overlap)}",
      flush=True)


def yoffs(tokens, enc, a):
    if enc == "C0_uniform":
        return [0.0, 0.0, 0.0]
    if enc == "C1_wordclass":
        return [ETA * a if t in verbs else 0.0 for t in tokens]
    return [ETA * a * pca01[t] for t in tokens]


_cache = {}


def simulate(ys, a):
    """Exact measurement distribution for a 3-atom chain with y-offsets."""
    key = (round(a, 6), tuple(round(y, 6) for y in ys))
    if key in _cache:
        return _cache[key]
    reg = Register({f"a{i}": ((i - 1) * a, ys[i]) for i in range(3)})
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("ryd", "rydberg_global")
    amp = InterpolatedWaveform(T, [0.0, OMEGA_MAX, OMEGA_MAX, 0.0])
    det = InterpolatedWaveform(T, [-DET_RANGE, 0.0, DET_RANGE])
    seq.add(Pulse(amp, det, 0.0), "ryd")
    res = QutipEmulator.from_sequence(seq).run()
    pr = np.abs(res.get_final_state().full().flatten()) ** 2
    meas = np.array([pr[7 ^ i] for i in range(8)])  # bit-convention flip
    meas = meas / meas.sum()
    _cache[key] = meas
    return meas


BITS = [format(i, "03b") for i in range(8)]


def features(meas):
    Pn = [float(sum(meas[i] for i in range(8) if BITS[i].count("1") == n))
          for n in range(4)]
    marg = [float(sum(meas[i] for i in range(8) if BITS[i][k] == "1"))
            for k in range(3)]
    corr = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        njk = float(sum(meas[i] for i in range(8)
                        if BITS[i][j] == "1" and BITS[i][k] == "1"))
        corr.append(njk - marg[j] * marg[k])
    return np.array(Pn + sorted(marg) + sorted(corr)), np.array(Pn)


def tvd(p, q):
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


# ── main sweep ───────────────────────────────────────────────────────────────
results = {
    "config": {"T_ns": T, "omega_max_rad_us": OMEGA_MAX,
               "det_range_rad_us": DET_RANGE, "eta": ETA,
               "spacings_um": [float(a) for a in SPACINGS],
               "n_shots": N_SHOTS,
               "n_seeds": N_SEEDS, "n_pairs": len(twins),
               "pca1_explained_variance": evr,
               "device": "AnalogDevice (pulser 1.8.0)",
               "note_dmm": "AnalogDevice has no DMM channels; word info "
                           "enters via geometry (perpendicular displacement)"},
    "V_nn_over_omega": {f"{a:.3f}": float(C6 / a ** 6 / OMEGA_MAX)
                        for a in SPACINGS},
    "C0_uniform": {"note": "SVO/VSO registers identical by construction "
                           "(all y=0, same chain): TVD == 0 exactly at every "
                           "spacing; verified geometrically."},
    "C1_wordclass": {}, "C2_embedding": {},
}

feat_store = {}
for si, a in enumerate(SPACINGS):
    ta = time.time()
    vno = C6 / a ** 6 / OMEGA_MAX
    # C1: two canonical registers (all SVO are N-V-N, all VSO are V-N-N)
    mA = simulate([0.0, ETA * a, 0.0], a)
    mB = simulate([ETA * a, 0.0, 0.0], a)
    _, PnA = features(mA)
    _, PnB = features(mB)
    results["C1_wordclass"][f"{a:.3f}"] = {
        "tvd_sym": tvd(PnA, PnB), "tvd_site": tvd(mA, mB)}
    # C2: all pairs
    X, y, groups, tv_sym, tv_site = [], [], [], [], []
    for pi, (i, j) in enumerate(twins):
        toksS, toksV = sents[i].split(), sents[j].split()
        mS = simulate(yoffs(toksS, "C2_embedding", a), a)
        mV = simulate(yoffs(toksV, "C2_embedding", a), a)
        _, PnS = features(mS)
        _, PnV = features(mV)
        empS = SHOT_RNG.multinomial(N_SHOTS, mS) / N_SHOTS
        empV = SHOT_RNG.multinomial(N_SHOTS, mV) / N_SHOTS
        fS, _ = features(empS)
        fV, _ = features(empV)
        X += [fS, fV]; y += [1, 0]; groups += [pi, pi]
        tv_sym.append(tvd(PnS, PnV)); tv_site.append(tvd(mS, mV))
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    feat_store[f"a_{a:.3f}"] = X
    aucs = []
    for seed in range(N_SEEDS):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y, groups):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                                         random_state=seed))])
            pipe.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], pipe.decision_function(X[te])))
    results["C2_embedding"][f"{a:.3f}"] = {
        "tvd_sym_median": float(np.median(tv_sym)),
        "tvd_sym_q25": float(np.percentile(tv_sym, 25)),
        "tvd_sym_q75": float(np.percentile(tv_sym, 75)),
        "tvd_site_median": float(np.median(tv_site)),
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "n_folds": len(aucs)}
    r1 = results["C1_wordclass"][f"{a:.3f}"]
    r2 = results["C2_embedding"][f"{a:.3f}"]
    print(f"[16] a={a:6.2f}um V/O={vno:9.4f} | C1 tvd={r1['tvd_sym']:.4f} | "
          f"C2 tvd_med={r2['tvd_sym_median']:.4f} "
          f"AUC={r2['auc_mean']:.3f}±{r2['auc_sd']:.3f} "
          f"({time.time() - ta:.0f}s, {len(_cache)} sims cached)", flush=True)
    json.dump(results, open("exp16_results.json", "w"), indent=2)

np.savez("exp16_features.npz",
         **feat_store, spacings=np.array(SPACINGS))

# ── figure ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vno = [C6 / a ** 6 / OMEGA_MAX for a in SPACINGS]
c1 = [results["C1_wordclass"][f"{a:.3f}"]["tvd_sym"] for a in SPACINGS]
c2m = [results["C2_embedding"][f"{a:.3f}"]["tvd_sym_median"] for a in SPACINGS]
c2lo = [results["C2_embedding"][f"{a:.3f}"]["tvd_sym_q25"] for a in SPACINGS]
c2hi = [results["C2_embedding"][f"{a:.3f}"]["tvd_sym_q75"] for a in SPACINGS]
auc = [results["C2_embedding"][f"{a:.3f}"]["auc_mean"] for a in SPACINGS]
aucsd = [results["C2_embedding"][f"{a:.3f}"]["auc_sd"] for a in SPACINGS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.plot(vno, c1, "o-", color="tab:red",
         label="C1 word-class (structure only, no embeddings)")
ax1.plot(vno, c2m, "s-", color="tab:blue",
         label="C2 embeddings (median over 60 pairs)")
ax1.fill_between(vno, c2lo, c2hi, color="tab:blue", alpha=0.2,
                 label="C2 IQR")
ax1.set_xscale("log"); ax1.set_xlabel(r"$V_{nn}/\Omega$ (interaction strength)")
ax1.set_ylabel("TVD between SVO and VSO twins (symmetrized)")
ax1.legend(fontsize=8); ax1.set_title("Grammatical distinguishability vs interaction")
ax2.errorbar(vno, auc, yerr=aucsd, fmt="o-", color="tab:green")
ax2.axhline(0.5, color="gray", ls="--", lw=1)
ax2.set_xscale("log"); ax2.set_xlabel(r"$V_{nn}/\Omega$")
ax2.set_ylabel("SVO-vs-VSO classifier AUC (C2)")
ax2.set_ylim(0.35, 1.02)
ax2.set_title("Word order from permutation-invariant statistics")
fig.suptitle("Exp16 Phase 0 — Arabic word order in analog neutral-atom dynamics "
             "(AnalogDevice, one global pulse, structure as geometry)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig_exp16_sweep.png", dpi=200, bbox_inches="tight")

results["runtime_sec"] = round(time.time() - t00, 1)
results["n_unique_sims"] = len(_cache)
json.dump(results, open("exp16_results.json", "w"), indent=2)
print(f"[16] DONE in {results['runtime_sec']}s, {len(_cache)} unique simulations",
      flush=True)
