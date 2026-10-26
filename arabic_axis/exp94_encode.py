# -*- coding: utf-8 -*-
"""exp94_encode.py -- doc 89: stories -> tensors via the ONE segmentation function (exp94_common). Grouped by K."""
import torch
import exp94_common as C


def encode(stories, angles):
    """-> dict K -> (batch dict, index list into `stories`)."""
    groups = {}
    for i, st in enumerate(stories):
        c = C.compile_story(st)
        groups.setdefault(len(c["cands"]), []).append((i, c, st))
    out = {}
    for K, rows in groups.items():
        t = lambda xs, dt=torch.long: torch.tensor(xs, dtype=dt)
        b = {"K": K,
             "cs": t([[C.STEMS.index(s) for s, _, _ in c["cands"]] for _, c, _ in rows]),
             "cg": t([[C.NOUN_G.index(g) for _, g, _ in c["cands"]] for _, c, _ in rows]),
             "cn": t([[C.NOUN_N.index(n) for _, _, n in c["cands"]] for _, c, _ in rows]),
             "theta": t([[angles[s] for s, _, _ in c["cands"]] for _, c, _ in rows], torch.float32),
             "theta_o": t([angles[c["obj"]] for _, c, _ in rows], torch.float32),
             "o": t([C.OBJECTS.index(c["obj"]) for _, c, _ in rows]),
             "v": t([C.VERBS.index(c["verb"][0]) for _, c, _ in rows]),
             "vg": t([C.VERB_G.index(c["verb"][1]) for _, c, _ in rows]),
             "vn": t([C.VERB_N.index(c["verb"][2]) for _, c, _ in rows]),
             "qv": t([C.VERBS.index(c["qverb"]) for _, c, _ in rows]),
             "x": t([c["x"] for _, c, _ in rows]),
             "gold": t([st["gold_agent"] for _, _, st in rows]),
             "label": t([st["label"] for _, _, st in rows])}
        out[K] = (b, [i for i, _, _ in rows])
    return out


def sub(b, j):
    return {k: (v[j] if torch.is_tensor(v) else v) for k, v in b.items()}
