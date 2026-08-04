"""Exp 25: robustness of published numbers to the parser-audit findings.
(a) word-order QFM L0/L1 and (b) tense L1, each on FULL set (reproduction)
vs EXCLUDING the fallback-parsed sentences (7 WO / 9 tense, audit_before.json).
Legacy (published) warmstart mode. If clean-subset numbers sit inside the
published CIs, the published results are robust to the parse-quality issue."""
import os, json, time
import numpy as np
import exp13_arabert_comparison as e

os.environ["QFM_WARMSTART"] = "legacy"
t0 = time.time()
audit = json.load(open("audit_before.json"))

def fb_set(split):
    ents = audit.get(split, {}).get("fallback_sentences", [])
    return {x["sentence"] if isinstance(x, dict) else x for x in ents}

data = json.load(open("sentences.json", encoding="utf-8"))
results = {}

def block(tag, sents, labels, L):
    circuits, vidx = e.build_circuits(sents, L, 1)
    vlabels = [labels[i] for i in vidx]
    means = []
    for seed in e.SEEDS:
        r = e.run_qfm_cv(sents, labels, L, seed, n_s_qubits=1, tag=tag,
                         circuits_cache=circuits, valid_labels_cache=vlabels)
        means.append(r["mean"])
    mu, sd = float(np.mean(means)), float(np.std(means))
    results[tag] = {"mean": mu, "sd": sd, "n_sentences": len(sents)}
    print(f"[25] {tag}: {mu:.4f} +/- {sd:.4f} (n={len(sents)})", flush=True)

wo = data["WordOrderMatched"]
fb_wo = fb_set("WordOrderMatched")
svo = [d for d in wo if d["label"] == "WordOrder_SVO"]
vso = [d for d in wo if d["label"] == "WordOrder_VSO"]
n = min(len(svo), len(vso))
s_full = [d["sentence"] for d in svo[:n]] + [d["sentence"] for d in vso[:n]]
l_full = ["WordOrder_SVO"] * n + ["WordOrder_VSO"] * n
pairs = [(s, l) for s, l in zip(s_full, l_full) if s not in fb_wo]
s_cln = [p[0] for p in pairs]
l_cln = [p[1] for p in pairs]
print(f"[25] WO fallbacks excluded: {len(s_full) - len(s_cln)}", flush=True)
for L in (0, 1):
    block(f"WO_full_L{L}", s_full, l_full, L)
    block(f"WO_clean_L{L}", s_cln, l_cln, L)

tn = data["TenseBinary"]
fb_tn = fb_set("TenseBinary")
ts = [d["sentence"] for d in tn]
tl = [d["label"] for d in tn]
tp = [(s, l) for s, l in zip(ts, tl) if s not in fb_tn]
print(f"[25] Tense fallbacks excluded: {len(ts) - len(tp)}", flush=True)
block("Tense_full_L1", ts, tl, 1)
block("Tense_clean_L1", [p[0] for p in tp], [p[1] for p in tp], 1)

results["runtime_sec"] = round(time.time() - t0, 1)
json.dump(results, open("results_exp25.json", "w"), indent=2)
print(f"[25] DONE in {results['runtime_sec']}s", flush=True)
