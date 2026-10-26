# -*- coding: utf-8 -*-
"""a3_fig3_twin_grouped.py -- doc 82 §13 A3 (registered 4b5a75d before running).
Published Fig.3 AraVec learning curve (exp13 run_learning_curves: N per class sampled at
random, so a test item's bag-twin is often in train) reproduced, then re-run with
TWIN-GROUPED sampling (N word-order pairs to train: both twins together; test = other
pairs). Pre-committed: twin-grouped ~0.5 (+-0.05) at every N => the published
'AraVec degrades with more data' trajectory is VOID as a finding."""
import os, json, hashlib
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
ROOT = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.join(ROOT, "archive", "02_discocat_paper")
kv = Word2Vec.load(os.path.join(ROOT, "aravec", "full_uni_cbow_300_twitter.mdl")).wv
def vec(w):
    base = w[2:] if w.startswith("ال") else w
    for c in (w, base):
        if kv.has_index_for(c): return kv.get_vector(c)
    return None
def s2v(s):
    v = [x for t in s.split() for x in [vec(t)] if x is not None]
    if not v: return np.zeros(kv.vector_size)
    m = np.mean(v, axis=0); n = np.linalg.norm(m); return m / n if n > 0 else m
d = json.load(open(os.path.join(PAPER, "sentences.json"), encoding="utf-8"))["WordOrderMatched"]
svo = [x["sentence"] for x in d if x["label"] == "WordOrder_SVO"][:60]; vso = [x["sentence"] for x in d if x["label"] == "WordOrder_VSO"][:60]
S = svo + vso; L = ["WordOrder_SVO"] * 60 + ["WordOrder_VSO"] * 60
y = LabelEncoder().fit(L).transform(L); X = np.stack([s2v(s) for s in S])
# twin groups by token multiset
bag = [tuple(sorted(s.split())) for s in S]
groups = defaultdict(list)
for i, b in enumerate(bag): groups[b].append(i)
pairs = [g for g in groups.values() if sorted(y[g].tolist()) == [0, 1] and len(g) == 2]
res = {"published_sampler": {}, "twin_grouped": {}, "analytic_0.5(1-N/60)": {}}
for N in (5, 10, 20, 40):
    pub, tg = [], []
    for trial in range(10):
        rng = np.random.default_rng(trial * 137)
        by = defaultdict(list)
        for i, l in enumerate(L): by[l].append(i)
        tr = np.concatenate([rng.choice(ix, size=N, replace=False) for ix in by.values()])
        te = np.array([i for i in range(len(S)) if i not in set(tr.tolist())])
        p = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=trial))]).fit(X[tr], y[tr])
        pub.append(accuracy_score(y[te], p.predict(X[te])))
        rng2 = np.random.default_rng(10_000 + trial)
        chosen = rng2.choice(len(pairs), size=min(N, len(pairs)), replace=False)
        tr2 = np.array([i for c in chosen for i in pairs[c]]); te2 = np.array([i for i in range(len(S)) if i not in set(tr2.tolist())])
        p2 = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=trial))]).fit(X[tr2], y[tr2])
        tg.append(accuracy_score(y[te2], p2.predict(X[te2])))
    res["published_sampler"][N] = {"mean": float(np.mean(pub)), "std": float(np.std(pub))}
    res["twin_grouped"][N] = {"mean": float(np.mean(tg)), "std": float(np.std(tg)), "n_pairs_available": len(pairs)}
    res["analytic_0.5(1-N/60)"][N] = 0.5 * (1 - N / 60)
    print("N=%2d published %.3f (analytic %.3f) | twin-grouped %.3f +- %.3f" % (N, np.mean(pub), 0.5 * (1 - N / 60), np.mean(tg), np.std(tg)))
res["recorded_learning_curves_AraVec"] = {5: 0.460, 10: 0.419, 20: 0.340, 40: 0.185}
res["a3_twin_grouped_all_within_0.05_of_0.5"] = all(abs(v["mean"] - 0.5) <= 0.05 for v in res["twin_grouped"].values())
print("a3 pre-committed reading fires (VOID):", res["a3_twin_grouped_all_within_0.05_of_0.5"])
OUT = os.path.join(ROOT, "results_a3_fig3_twin_grouped.json"); assert not os.path.exists(OUT)
json.dump(res, open(OUT, "w"), indent=1)
