"""Dump AraVec vectors for every unique WordOrderMatched word -> exp16_wordvecs.json.
Feeds Exp16 (analog/Pulser) Encoding A detunings on the Mac without needing the
2.9GB model there. Loads gensim directly (no AraBERT/lambeq overhead).
"""
import json
from gensim.models import Word2Vec

data = json.load(open("sentences.json", encoding="utf-8"))["WordOrderMatched"]
words = sorted({w for d in data for w in d["sentence"].split()})
m = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl")
kv = m.wv

out, missing = {}, []
for w in words:
    cands = [w] + ([w[2:]] if w.startswith("ال") else [])
    vec = None
    for c in cands:
        try:
            if c in kv:
                vec = [float(x) for x in kv[c]]
                break
        except KeyError:
            pass
    if vec is None:
        missing.append(w)
    else:
        out[w] = vec

json.dump({"dim": 300, "vectors": out, "missing": missing},
          open("exp16_wordvecs.json", "w"), ensure_ascii=False)
print(f"dumped {len(out)}/{len(words)} words; missing: {missing}")
