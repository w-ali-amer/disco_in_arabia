# -*- coding: utf-8 -*-
"""exp42_make_embeddings.py -- build the FROZEN 2D noun embeddings for exp42.

One-off tool (run once, before any training; artifact committed to git):
reads the exp41 referent inventory, looks the nouns up in the repo's
existing AraVec model (aravec/full_uni_cbow_300_twitter.mdl, gensim 4.3.3
in the venv -- coverage verified 44/44 with ta-marbuta normalization
ta' marbuta U+0629 -> ha' U+0647, AraVec's own preprocessing convention),
projects 300d -> 2D by PCA over the 44 noun vectors, unit-normalizes, and
writes exp42_noun_embeddings.json with a canonical sha256.

Doc 22 SS3.1: embeddings are initialization ONLY -- frozen, never targets.
Names carry no answer signal by construction (exp41 V3 balance audit), so
these vectors only break symmetry between referent wires.

Fallback (--random): deterministic per-noun angles derived from
sha256(noun) -- no model needed, fully auditable. The chosen source is
recorded in the artifact's meta and in every results JSON.

Intro convention (consumed by exp42_compiler): angle = atan2(y, x) of the
unit 2D vector; wire prepared as Ry(angle)|0>.
"""

import argparse
import hashlib
import json
import math

DATA = "stories_exp41.json"
OUT = "exp42_noun_embeddings.json"
ARAVEC = "aravec/full_uni_cbow_300_twitter.mdl"


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def norm_ta_marbuta(w):
    return w.replace("ة", "ه")


def aravec_2d(nouns):
    import numpy as np
    from gensim.models import Word2Vec

    wv = Word2Vec.load(ARAVEC).wv
    rows = []
    for n in nouns:
        key = n if n in wv.key_to_index else norm_ta_marbuta(n)
        if key not in wv.key_to_index:
            raise KeyError("noun %r not in AraVec even after ta-marbuta "
                           "normalization -- use --random instead" % n)
        rows.append(np.asarray(wv[key], dtype=np.float64))
    M = np.stack(rows)
    M = M - M.mean(axis=0, keepdims=True)
    # PCA via SVD; deterministic given identical inputs/libs, and the
    # resulting vectors are FROZEN in the artifact anyway (hash recorded).
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    P = M @ Vt[:2].T                      # (44, 2)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return {n: [float(x), float(y)] for n, (x, y) in zip(nouns, P)}


def random_frozen_2d(nouns):
    out = {}
    for n in nouns:
        h = hashlib.sha256(("exp42-frozen-noun:" + n).encode("utf-8")).digest()
        a = (int.from_bytes(h[:8], "big") / 2 ** 64) * 2.0 * math.pi
        out[n] = [math.cos(a), math.sin(a)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random", action="store_true",
                    help="hash-derived frozen vectors instead of AraVec")
    args = ap.parse_args()

    with open(DATA, encoding="utf-8") as f:
        meta = json.load(f)["meta"]
    nouns = list(meta["humans"]) + list(meta["objects"])

    if args.random:
        vecs, source = random_frozen_2d(nouns), "random_frozen_sha256_per_noun"
    else:
        vecs, source = aravec_2d(nouns), (
            "aravec/full_uni_cbow_300_twitter.mdl + ta-marbuta normalization "
            "+ PCA-2D over the 44 referent vectors + unit norm")

    emb = {n: {"vec2": v, "angle": math.atan2(v[1], v[0])}
           for n, v in vecs.items()}
    artifact = {
        "meta": {
            "experiment": "exp42",
            "purpose": "frozen 2D noun intro embeddings (doc 22 SS3.1; "
                       "initialization only, never training targets)",
            "source": source,
            "dataset_sha256": meta["dataset_sha256"],
            "intro_convention": "wire prepared as Ry(angle)|0>, "
                                "angle = atan2(y, x)",
            "n_nouns": len(emb),
        },
        "embeddings": emb,
    }
    artifact["embeddings_sha256"] = canonical_sha256(emb)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[exp42] wrote %s  source=%s" % (OUT, source))
    print("[exp42] embeddings sha256 = %s" % artifact["embeddings_sha256"])


if __name__ == "__main__":
    main()
