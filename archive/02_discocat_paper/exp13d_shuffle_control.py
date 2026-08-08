"""Exp 13d: shuffled-embedding control for the depth x content finding.

exp13b/c showed: hash parameters decay with depth (64.9 -> 56.7 by L4) while
AraVec parameters stay flat (62-65). CONFOUND: AraVec phases (v+1)*pi
concentrate near pi, hash phases are uniform on [0, 2pi) — "content
alignment" and "phase distribution" are entangled.

Control: permute the word->vector assignment (fixed seed derangement over the
dataset vocabulary). Preserves the phase distribution exactly; destroys
word-content alignment. Word-order L1..L4, same protocol.
- If shuffled ~ fixed: the stability comes from the PHASE DISTRIBUTION, not
  from content alignment (content story dies, distribution story wins).
- If shuffled decays like hash: content alignment itself stabilizes depth.
"""
import os, json, time
import numpy as np
import exp13_arabert_comparison as e

OUT_PATH = "results_exp13d.json"
t00 = time.time()
data = json.load(open("sentences.json", encoding="utf-8"))
wo = data["WordOrderMatched"]
svo = [d for d in wo if d["label"] == "WordOrder_SVO"]
vso = [d for d in wo if d["label"] == "WordOrder_VSO"]
n = min(len(svo), len(vso))
sents = [d["sentence"] for d in svo[:n]] + [d["sentence"] for d in vso[:n]]
labels = ["WordOrder_SVO"] * n + ["WordOrder_VSO"] * n

# fixed-seed derangement over dataset base vocabulary
vocab = sorted({w for d in wo for w in d["sentence"].split()})
rng = np.random.default_rng(123)
perm = list(vocab)
while True:
    rng.shuffle(perm)
    if all(a != b for a, b in zip(vocab, perm)):
        break
SHUF = dict(zip(vocab, perm))
print(f"[13d] derangement over {len(vocab)} words", flush=True)

_orig_aravec = e._aravec_vec
e._aravec_vec = lambda w: _orig_aravec(SHUF.get(w, w))

os.environ["QFM_WARMSTART"] = "fixed"   # patched lookup + shuffled assignment
results = {"note": "fixed-mode warmstart with deranged word->vector map "
                   "(seed 123); compare to exp13b/c fixed and legacy"}
for L in (1, 2, 3, 4):
    circuits, vidx = e.build_circuits(sents, L, 1)
    vlabels = [labels[i] for i in vidx]
    t0 = time.time()
    seed_means = []
    for seed in e.SEEDS:
        r = e.run_qfm_cv(sents, labels, L, seed, n_s_qubits=1,
                         tag="QFM_shuffled", circuits_cache=circuits,
                         valid_labels_cache=vlabels)
        seed_means.append(r["mean"])
    mu, sd = float(np.mean(seed_means)), float(np.std(seed_means))
    results[f"shuffled_L{L}"] = {"mean": mu, "std_across_seeds": sd,
                                 "seed_means": seed_means,
                                 "runtime_sec": round(time.time() - t0, 1)}
    print(f"[13d] shuffled_L{L}: {mu:.4f} +/- {sd:.4f}", flush=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)

# comparison table
try:
    b = json.load(open("results_exp13b.json"))
    c = json.load(open("results_exp13c.json"))
    print("\n[13d] L | fixed   | shuffled | legacy", flush=True)
    for L in (1, 2, 3, 4):
        fx = (b.get(f"fixed_L{L}") or c.get(f"wordorder_fixed_L{L}"))["mean"]
        lg = (b.get(f"legacy_L{L}") or c.get(f"wordorder_legacy_L{L}"))["mean"]
        sh = results[f"shuffled_L{L}"]["mean"]
        print(f"[13d] {L} | {fx:.4f} | {sh:.4f}  | {lg:.4f}", flush=True)
except Exception as ex:
    print(f"[13d] comparison skipped: {ex}", flush=True)

results["total_runtime_sec"] = round(time.time() - t00, 1)
json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"[13d] DONE in {results['total_runtime_sec']}s", flush=True)
