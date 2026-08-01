"""Re-dump AraVec vectors with AraVec-style normalization (ة→ه, ى→ي, أإآ→ا)."""
import json
from gensim.models import Word2Vec

def norm(w):
    return (w.replace("ة", "ه").replace("ى", "ي")
             .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا"))

data = json.load(open("sentences.json", encoding="utf-8"))["WordOrderMatched"]
words = sorted({w for d in data for w in d["sentence"].split()})
kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv

out, missing = {}, []
for w in words:
    cands = []
    for base in (w, w[2:] if w.startswith("ال") else w):
        cands += [base, norm(base)]
    vec = None
    for c in dict.fromkeys(cands):
        if c in kv:
            vec = [float(x) for x in kv[c]]
            break
    (out.__setitem__(w, vec) if vec is not None else missing.append(w))
json.dump({"dim": 300, "vectors": out, "missing": missing},
          open("exp16_wordvecs.json", "w"), ensure_ascii=False)
print(f"dumped {len(out)}/{len(words)}; missing: {missing}")
