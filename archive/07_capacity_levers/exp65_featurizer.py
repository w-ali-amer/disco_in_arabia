# -*- coding: utf-8 -*-
"""exp65_featurizer.py -- doc 51 SS4: the MORPHOLOGY-AWARE angle map.

The featurizer (n-gram vocabulary AND PCA basis) is fit on the TRAINING
inventory only, then APPLIED to held-out nouns. A held-out noun's angle is
COMPUTED from its surface form by the frozen featurizer, never looked up.
Fitting on all nouns -- as tonight's diagnostic did -- would be leakage.
"""
import math, random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import exp41_story_generator as gen

SPLIT_SEED = 65000                      # doc 51 SS5, frozen BEFORE the bound
FAAIL = {"الطالب","العامل","الكاتب","الحارس","اللاعب","السائق","الضابط","الباحث","القائد"}
N_TRAIN_FAAIL, N_TRAIN_OTHER = 6, 10


def strip_al(w):
    return w[2:] if w.startswith("ال") else w


def split_inventory(seed=SPLIT_SEED):
    H = sorted(gen.HUMANS)
    f = sorted(FAAIL); o = sorted(set(H) - FAAIL)
    rng = random.Random(seed)
    rng.shuffle(f); rng.shuffle(o)
    train = f[:N_TRAIN_FAAIL] + o[:N_TRAIN_OTHER]
    test = f[N_TRAIN_FAAIL:] + o[N_TRAIN_OTHER:]
    assert not (set(train) & set(test))
    return sorted(train), sorted(test)


def fit_featurizer(train_nouns, scramble=False):
    """Fit n-gram vocab + PCA basis on TRAIN ONLY. Returns a transform closure."""
    def prep(w):
        s = strip_al(w)
        if scramble:                     # doc 51 SS6: destroy templatic order,
            c = list(s)                  # preserve character inventory
            random.Random(hash(s) & 0xffff).shuffle(c)
            s = "".join(c)
        return s
    # doc 51 SS11.1: POSITIONAL features. faail is C-alif-C-C and faaal is
    # C-C-alif-C, so the alif's POSITION discriminates them; a bag-of-characters
    # map is blind to exactly that, which is why SS10's scramble control passed.
    def tok(w):
        c = list(prep(w))
        return " ".join("%d:%s" % (i, ch) for i, ch in enumerate(c)) + " LEN%d" % len(c)
    vec = CountVectorizer(analyzer="word", token_pattern=r"\S+")
    X = vec.fit_transform([tok(w) for w in train_nouns]).toarray().astype(float)
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    basis = Vt[:2]

    def angle_of(word):
        x = vec.transform([tok(word)]).toarray().astype(float) - mu
        p = (x @ basis.T)[0]
        n = np.linalg.norm(p)
        if n < 1e-12:
            return math.pi                       # degenerate -> centre
        p = p / n
        return math.atan2(p[1], p[0])
    return angle_of


def angles_for(train_nouns, all_nouns, scramble=False):
    f = fit_featurizer(train_nouns, scramble=scramble)
    return {w: float(f(w)) for w in all_nouns}
