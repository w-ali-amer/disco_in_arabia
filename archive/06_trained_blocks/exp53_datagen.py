# -*- coding: utf-8 -*-
"""exp53_datagen.py -- doc 37, the root x pattern rung (RP).

THE ARABIC BET. Arabic verbs are a consonantal ROOT cast into a PATTERN.
A model with one free block per verb has learned twelve unrelated symbols;
a model that COMPOSES root and pattern has learned the morphology. The
distinguishing test is not accuracy on seen verbs -- exp51 already showed
those are learnable as atoms -- it is GENERALISATION TO UNSEEN CELLS.

Grid signed off 2026-08-15 after screening 24 roots against CAMeL's
morphological database (13 had complete II/III/IV paradigms; filtered by
whether all three cells take a DIRECT human object):

            II فعّل      III فاعل     IV أفعل
  خ.ب.ر     خبّر         خابر         أخبر
  ش.غ.ل     شغّل         شاغل         أشغل
  س.ك.ن     سكّن         ساكن         أسكن
  ر.ج.ع     رجّع         راجع         أرجع

HELD-OUT CELLS (frozen, doc 37 SS6.1): one per root, cycling patterns --
(خ.ب.ر,II), (ش.غ.ل,III), (س.ك.ن,IV), (ر.ج.ع,II). Every root appears in
training in its other patterns and every pattern appears in training with
other roots, so a compositional model has all the parts while an atomic
model has never seen the target block.

ITEM ROUTING (frozen): train/val draw both core verbs AND all fillers from
SEEN cells. Test places one HELD verb and one SEEN verb on the two core
pairs, draws fillers from SEEN, and the question targets the HELD verb's
pair -- the discriminative case.

Items reuse exp51's verb-exchange twin construction verbatim (doc 35 SS2):
twins differ ONLY in which verb attaches to which argument pair, they are
bag-identical, and a tied-block model compiles both to the identical
statevector and is pinned at exactly 0.500.
"""

import hashlib
import json
import random
from collections import Counter

import exp41_story_generator as gen

DESIGN_DOC = ("qnlp_private_docs/37_exp53_root_pattern_composition.md "
              "(1ba1469 criteria + 38a9fd4 amendment + b52d1f9 grid, FROZEN)")
RUNG = "RP"
SEED = 53000
K, T = 4, 5
SPLIT_PAIRS = {"train": 600, "val": 150, "test": 200}

ROOTS = ["خ.ب.ر", "ش.غ.ل", "س.ك.ن", "ر.ج.ع"]
PATTERNS = ["II", "III", "IV"]
GRID = {
    ("خ.ب.ر", "II"): "خبّر",  ("خ.ب.ر", "III"): "خابر", ("خ.ب.ر", "IV"): "أخبر",
    ("ش.غ.ل", "II"): "شغّل",  ("ش.غ.ل", "III"): "شاغل", ("ش.غ.ل", "IV"): "أشغل",
    ("س.ك.ن", "II"): "سكّن",  ("س.ك.ن", "III"): "ساكن", ("س.ك.ن", "IV"): "أسكن",
    ("ر.ج.ع", "II"): "رجّع",  ("ر.ج.ع", "III"): "راجع", ("ر.ج.ع", "IV"): "أرجع",
}
HELD_CELLS = [("خ.ب.ر", "II"), ("ش.غ.ل", "III"),
              ("س.ك.ن", "IV"), ("ر.ج.ع", "II")]
HELD = [GRID[c] for c in HELD_CELLS]
SEEN = [v for c, v in GRID.items() if c not in HELD_CELLS]
CELL_OF = {v: c for c, v in GRID.items()}


def canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_pair(rng, split, pair_no, v1, v2, filler_pool):
    """exp51 verb-exchange twin (doc 35 SS2), with v1/v2 on the two core
    pairs at the same slots and the question targeting v1's pair."""
    cast = rng.sample(gen.HUMANS, K)
    a, x, c, d = cast
    slots = sorted(rng.sample(range(T), 2))
    s1, s2 = slots
    others = [v for v in filler_pool if v not in (v1, v2)]
    fillers, used = [], set()
    while len(fillers) < T - 2:
        fv = rng.choice(others)
        fa, fp = rng.sample(cast, 2)
        if (fv, fa, fp) in used:
            continue
        used.add((fv, fa, fp))
        fillers.append({"verb": fv, "agent": fa, "patient": fp,
                        "pro_drop": False})

    def assemble(c1, c2):
        ev, it = [None] * T, iter(fillers)
        ev[s1], ev[s2] = c1, c2
        return [e if e is not None else next(it) for e in ev]

    yes = assemble({"verb": v1, "agent": a, "patient": x, "pro_drop": False},
                   {"verb": v2, "agent": c, "patient": d, "pro_drop": False})
    no = assemble({"verb": v2, "agent": a, "patient": x, "pro_drop": False},
                  {"verb": v1, "agent": c, "patient": d, "pro_drop": False})
    question = {"type": "Q0", "surface": "هل %s %s %s" % (v1, a, x),
                "v": v1, "a": a, "p": x}
    referents = [{"name": h, "type": "human"} for h in cast]
    out = []
    for events, ans in ((yes, "yes"), (no, "no")):
        sents, flags = gen.render(rng, events)
        out.append({"answer": ans, "events": events,
                    "sentences": sents, "order_flags": flags})
    rng.shuffle(out)
    pid = "p%05d" % pair_no
    return [{
        "id": "%s_%s_%s" % (split, pid, "ab"[j]), "pair_id": pid,
        "split": split, "qtype": "RP", "hop_count": 1,
        "pro_drop_item": False, "order_dependent": False,
        "answer": p["answer"], "K": K, "T": T, "flip_idx": [s1, s2],
        "referents": referents, "events": p["events"],
        "sentences": p["sentences"], "order_flags": p["order_flags"],
        "question": question,
        "queried_verb": v1, "queried_cell": list(CELL_OF[v1]),
        "queried_is_held_out": v1 in HELD,
    } for j, p in enumerate(out)]


def generate():
    rng = random.Random(SEED)
    items, n = [], 0
    for split, npairs in SPLIT_PAIRS.items():
        for _ in range(npairs):
            n += 1
            if split == "test":
                v1 = HELD[n % len(HELD)]                # question targets HELD
                v2 = rng.choice([v for v in SEEN
                                 if CELL_OF[v][0] != CELL_OF[v1][0]])
            else:
                v1, v2 = rng.sample(SEEN, 2)
            items.extend(build_pair(rng, split, n, v1, v2, SEEN))

    # ---- invariants
    by = {}
    for it in items:
        by.setdefault(it["pair_id"], []).append(it)
    for pid, (p, q) in by.items():
        bp = Counter((" . ".join(p["sentences"]) + " "
                      + p["question"]["surface"]).split())
        bq = Counter((" . ".join(q["sentences"]) + " "
                      + q["question"]["surface"]).split())
        assert bp == bq, "twin bag mismatch %s" % pid
        assert {p["answer"], q["answer"]} == {"yes", "no"}, pid
        for it in (p, q):
            qq = it["question"]
            hit = any(e["verb"] == qq["v"] and e["agent"] == qq["a"]
                      and e["patient"] == qq["p"] for e in it["events"])
            assert hit == (it["answer"] == "yes"), it["id"]
            if it["split"] != "test":
                assert all(e["verb"] in SEEN for e in it["events"]), \
                    "HELD verb leaked into %s (%s)" % (it["split"], it["id"])
            else:
                assert it["queried_is_held_out"], it["id"]
        s1, s2 = p["flip_idx"]
        for i in range(T):
            if i in (s1, s2):
                assert p["events"][i]["verb"] != q["events"][i]["verb"], pid
            else:
                assert p["events"][i] == q["events"][i], pid

    sha = canonical_sha256(items)
    ev = Counter(e["verb"] for it in items for e in it["events"])
    meta = {
        "experiment": "exp53", "rung": RUNG, "design_doc": DESIGN_DOC,
        "seed": SEED, "dataset_sha256": sha, "n_items": len(items),
        "split_counts": dict(Counter(it["split"] for it in items)),
        "answer_balance": dict(Counter(it["answer"] for it in items)),
        "grid": {"%s|%s" % k: v for k, v in GRID.items()},
        "roots": ROOTS, "patterns": PATTERNS,
        "held_out_cells": [list(c) for c in HELD_CELLS],
        "held_out_verbs": HELD, "seen_verbs": SEEN,
        "verb_inventory": {v: {"human_patients": True, "objects": [],
                               "swap": True} for v in GRID.values()},
        "cell_of": {v: list(c) for v, c in CELL_OF.items()},
        "event_verb_counts": dict(ev), "humans": gen.HUMANS, "objects": [],
        "holdout_verb_argument_pairs": [],
        "grid_provenance": (
            "screened against CAMeL MorphologyDB on WSL: 24 roots analysed, 13 "
            "with a complete II/III/IV verb paradigm, filtered to those whose "
            "three cells all take a DIRECT human object; native-speaker "
            "sign-off 2026-08-15, no substitutions (doc 37 SS6.1)"),
        "invariants_asserted": [
            "twin bag-of-words identity", "answers exactly {yes,no}",
            "label follows from events in BOTH twins",
            "twins differ ONLY in the verb assignment on the two core slots",
            "NO held-out verb appears anywhere in train/val",
            "every test item queries a HELD-OUT cell",
        ],
    }
    out = "stories_exp53_%s.json" % RUNG
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False)
    print("[exp53] wrote %s" % out)
    print("[exp53] n=%d splits=%s balance=%s"
          % (len(items), meta["split_counts"], meta["answer_balance"]))
    print("[exp53] sha256=%s" % sha)
    print("[exp53] SEEN(%d)=%s" % (len(SEEN), SEEN))
    print("[exp53] HELD(%d)=%s" % (len(HELD), HELD))
    print("[exp53] event verb counts: %s" % dict(ev))
    print("[exp53] all invariants asserted OK")
    return meta


if __name__ == "__main__":
    generate()
