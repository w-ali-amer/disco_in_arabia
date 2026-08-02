"""Exp 13c: (a) does circuit depth help ONLY when parameters carry content?
(b) does QFM-with-real-embeddings start seeing lexical (tense) signal?

Motivated by exp13b's interaction pattern: legacy (hash) L1->L2 = 64.9->61.8
(depth hurts) vs fixed (AraVec) L1->L2 = 63.6->64.5 (depth stops hurting).
Blocks: word-order L3/L4 x {fixed, legacy}; tense L1/L2 x {fixed, legacy}
(legacy tense L1 doubles as a reproduction check vs the paper's 56.0%).
Also computes the paired seed-level L2-L1 test from results_exp13b.json.
Protocol identical to exp13 Task A/B (SEEDS x StratifiedKFold(N_FOLDS)).
"""
import os, json, time
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import exp13_arabert_comparison as e

OUT_PATH = "results_exp13c.json"
t00 = time.time()
data = json.load(open("sentences.json", encoding="utf-8"))

wo = data["WordOrderMatched"]
svo = [d for d in wo if d["label"] == "WordOrder_SVO"]
vso = [d for d in wo if d["label"] == "WordOrder_VSO"]
n = min(len(svo), len(vso))
wo_sents = [d["sentence"] for d in svo[:n]] + [d["sentence"] for d in vso[:n]]
wo_labels = ["WordOrder_SVO"] * n + ["WordOrder_VSO"] * n

tense = data["TenseBinary"]
t_sents = [d["sentence"] for d in tense]
t_labels = [d["label"] for d in tense]
print(f"[13c] WO {len(wo_sents)} sents; Tense {len(t_sents)} sents", flush=True)

results = {}

# paired L2-vs-L1 test from exp13b (same seeds -> same fold splits)
try:
    b = json.load(open("results_exp13b.json"))
    for mode in ("fixed", "legacy"):
        d1 = np.array(b[f"{mode}_L1"]["seed_means"])
        d2 = np.array(b[f"{mode}_L2"]["seed_means"])
        t, pt = ttest_rel(d2, d1)
        try:
            w, pw = wilcoxon(d2, d1)
        except ValueError:
            pw = 1.0
        results[f"paired_L2_minus_L1_{mode}"] = {
            "mean_diff": float(np.mean(d2 - d1)),
            "ttest_p": float(pt), "wilcoxon_p": float(pw),
            "seed_diffs": [float(x) for x in (d2 - d1)]}
        print(f"[13c] paired L2-L1 ({mode}): {np.mean(d2-d1):+.4f} "
              f"t-p={pt:.4f} w-p={pw:.4f}", flush=True)
except FileNotFoundError:
    print("[13c] results_exp13b.json not found; skipping paired test", flush=True)


def run_block(task, sents, labels, L):
    circuits, vidx = e.build_circuits(sents, L, 1)
    vlabels = [labels[i] for i in vidx]
    for mode in ("fixed", "legacy"):
        os.environ["QFM_WARMSTART"] = mode
        t0 = time.time()
        seed_means, folds_all = [], []
        for seed in e.SEEDS:
            r = e.run_qfm_cv(sents, labels, L, seed, n_s_qubits=1,
                             tag=f"{task}_{mode}", circuits_cache=circuits,
                             valid_labels_cache=vlabels)
            seed_means.append(r["mean"]); folds_all += r["folds"]
        mu, sd = float(np.mean(seed_means)), float(np.std(seed_means))
        key = f"{task}_{mode}_L{L}"
        results[key] = {"mean": mu, "std_across_seeds": sd,
                        "seed_means": seed_means,
                        "runtime_sec": round(time.time() - t0, 1)}
        print(f"[13c] {key}: {mu:.4f} +/- {sd:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)


for L in (3, 4):
    run_block("wordorder", wo_sents, wo_labels, L)
for L in (1, 2):
    run_block("tense", t_sents, t_labels, L)

results["total_runtime_sec"] = round(time.time() - t00, 1)
json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"[13c] DONE in {results['total_runtime_sec']}s", flush=True)
