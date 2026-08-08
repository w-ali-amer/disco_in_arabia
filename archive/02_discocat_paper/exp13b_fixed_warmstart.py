"""Exp 13b: word-order QFM re-run with the repaired AraVec warmstart.

Completes the ERRATUM: reports QFM L0/L1/L2 on Task A (WordOrderMatched,
60/class) under (a) mode=fixed — real AraVec-derived parameters (~81% symbol
hit rate) — and (b) mode=legacy — the published hash-fallback behaviour, as a
live reproduction check against the paper's numbers (L1 = 64.9%).

Protocol identical to exp13 Task A: per seed, StratifiedKFold(N_FOLDS) CV,
NumpyModel probability features -> StandardScaler -> SVM-RBF (C=10), over
exp13.SEEDS. Circuits built once per layer count and shared across modes.

Pre-registered expectation: fixed-mode L1 lands near 64.9% / topology-only
64.2% (word content adds little on a task that is topological by design).
A large deviation in either direction is itself informative and goes in the
journal version.
"""
import os, json, time
import numpy as np
import exp13_arabert_comparison as e

OUT_PATH = "results_exp13b.json"
t00 = time.time()

data = json.load(open("sentences.json", encoding="utf-8"))
wo = data["WordOrderMatched"]
svo = [d for d in wo if d["label"] == "WordOrder_SVO"]
vso = [d for d in wo if d["label"] == "WordOrder_VSO"]
n = min(len(svo), len(vso))
sents = [d["sentence"] for d in svo[:n]] + [d["sentence"] for d in vso[:n]]
labels = ["WordOrder_SVO"] * n + ["WordOrder_VSO"] * n
print(f"[13b] Task A: {len(sents)} sentences, {n}/class", flush=True)

results = {"protocol": f"{e.N_SEEDS} seeds x {e.N_FOLDS}-fold stratified CV, "
                       "SVM-RBF on NumpyModel probability features"}
rng = np.random.default_rng(7)

for L in (0, 1, 2):
    circuits, valid_idx = e.build_circuits(sents, L, 1)
    vlabels = [labels[i] for i in valid_idx]
    print(f"[13b] L{L}: {len(circuits)} circuits", flush=True)
    for mode in ("fixed", "legacy"):
        os.environ["QFM_WARMSTART"] = mode
        t0 = time.time()
        seed_means, folds_all = [], []
        for seed in e.SEEDS:
            r = e.run_qfm_cv(sents, labels, L, seed, n_s_qubits=1,
                             tag=f"QFM_{mode}",
                             circuits_cache=circuits,
                             valid_labels_cache=vlabels)
            seed_means.append(r["mean"])
            folds_all += r["folds"]
        mu = float(np.mean(seed_means))
        sd = float(np.std(seed_means))
        boots = [float(np.mean(rng.choice(folds_all, size=len(folds_all),
                                          replace=True)))
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        key = f"{mode}_L{L}"
        results[key] = {"mean": mu, "std_across_seeds": sd,
                        "ci95_bootstrap_folds": [float(lo), float(hi)],
                        "seed_means": seed_means,
                        "n_folds_total": len(folds_all),
                        "runtime_sec": round(time.time() - t0, 1)}
        print(f"[13b] {key}: {mu:.4f} +/- {sd:.4f}  CI95 [{lo:.3f},{hi:.3f}]  "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

results["total_runtime_sec"] = round(time.time() - t00, 1)
json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"[13b] DONE in {results['total_runtime_sec']}s", flush=True)
