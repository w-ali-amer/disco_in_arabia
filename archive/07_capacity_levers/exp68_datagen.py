# -*- coding: utf-8 -*-
"""exp68_datagen.py -- doc 54 SS6.1: a NEW SB dataset for the WIDTH lever.

WHY A NEW DATASET (doc 54 SS6.1, and it is worth restating at the top of the
file that builds it). exp61 reused `stories_exp60_SB.json` untouched because it
changed only the ANGLE MAP, and doc 47 SS5's argument was that "the labels depend
only on which nouns are bearers -- which does not change". THAT ARGUMENT DOES
NOT TRANSFER. exp60's bearer set is 12 nouns chosen by the datagen seed (half of
each referent inventory, for balance); exp68's bearer set is a REAL SEMANTIC
CATEGORY of four nouns (doc 54 SS5.2, user-confirmed 2026-08-23). The labels
therefore DO change, and reusing exp60's items would silently measure exp60's
arbitrary 12-way split.

WHAT IS REUSED, EXACTLY (doc 54 SS6.1): the SB construction itself (doc 46 SS5)
-- one event, two-referent cast, exactly one bearer, twins exchanging
agent/patient over an identical cast, identical question surface, opposite
labels; the 6-verb inventory from `stories_exp51_VT.json`; the per-pair cast
shuffle; one rendering per pair. V1 remains EXACTLY 0.500 by construction and is
re-asserted in `exp68_validate.py` rather than inherited.

THE SPLIT IS THE ONE THING THAT CHANGES (doc 54 SS6.1.1), and it is frozen:

    casts       = { (b, n) : b in B, n in N }        |B| = 4, |N| = 20  -> 80
    items       = casts x 6 verbs x 2 twins                            -> 960
    test        = 25 casts at the registered seed    -> 150 pairs      -> 300
    train + val = the remaining 55 casts, split 80/20 BY CAST (never within
                  a pair)                            -> 44 / 11 casts

HELD-OUT MEANS UNSEEN PAIRINGS, NOT UNSEEN NOUNS. That is a WEAKER claim than
exp60/exp61 made, and the weakening is deliberate and is the price of doc 54
SS4.3's legitimate inventory-wide PCA fit: with a four-noun bearer set a
held-out-REFERENT split would put one or two bearers in the test inventory and
there would be no usable test size at all. doc 54 SS10 forbids reporting exp68
as novel-filler generalisation, and `exp68_validate.py` records the consequence
rather than hiding it: under a pairing split a TOKEN-IDENTITY model transfers to
test by design, so exp60's V2-H and V6 held-out BARS cannot be inherited (they
would be failing a dataset for doing exactly what it was designed to do). The
circuit never sees a token -- it consumes only per-referent angles -- so the
shortcut is available to surface baselines and to nothing exp68 measures.

COVERAGE IS ASSERTED, NOT HOPED FOR (doc 54 SS6.1.1, exp61 SS5's band-coverage
assertion transposed): every bearer AND every non-bearer must appear in at least
one TRAIN cast, or "unseen pairing" degenerates into "unseen noun" for that
referent and the split stops measuring what it claims. A uniform draw fails this
about 40% of the time here (each non-bearer sits in only 4 of the 80 casts), so
the generator REDRAWS from the registered seed's own stream until coverage
holds, and records how many draws that took. The seed is registered; the
acceptance rule is registered; nothing is chosen after the fact.

n_test = 300 is deliberately identical to exp60/exp61's so that doc 54 SS7's
0.06 threshold -- 2 x the binomial SD at n = 300 -- transfers EXACTLY rather
than approximately.

MEMBERSHIP IS IMPORTED, NEVER RETYPED. The bearer set comes from
`instrument_sb_bounds.CATEGORIES["nonoccupational"]`, the committed module whose
bounds doc 54 SS12 cites, so the dataset and the bounds cannot drift apart.

    python exp68_datagen.py                  # stories_exp68_WN.json, seed 68000
    python exp68_datagen.py --seed 69000     # doc 54 SS6.1.1's CONFIRM draw
"""
import argparse, json, random, sys
from collections import Counter

import exp41_story_generator as gen
import exp42_compiler as comp
import instrument_sb_bounds as INS

DESIGN_DOC = "qnlp_private_docs/54_exp68_width_lever.md (1025d21, FROZEN)"
VT_SRC = "stories_exp51_VT.json"          # verb inventory provenance
OUT = "stories_exp68_WN.json"             # WN = Width / Non-occupational
SEED = 68000                              # registered; confirm draw = SEED + 1000
CONFIRM_SEED = SEED + 1000                # doc 46 SS10.4's convention, doc 54 SS6.1.1
CATEGORY = "nonoccupational"              # doc 54 SS6.3, FROZEN target
N_TEST_CASTS = 25                         # doc 54 SS6.1.1, frozen -> 300 test items
VAL_FRACTION = 0.20                       # by CAST, never within a pair
MARKER = "المميز"                          # placeholder; names no referent
MAX_DRAWS = 500                           # coverage redraw budget; exceeding it is a STOP


def render_pair(rng):
    """One rendering per pair; both twins share the order flag, so the only
    surface difference inside a pair is the noun swap (doc 46 SS5)."""
    if rng.random() < 0.5:
        return "VSO", (lambda a, p: "%s %s %s" % (VERB_SLOT[0], a, p))
    return "SVO", (lambda a, p: "%s %s %s" % (a, VERB_SLOT[0], p))


VERB_SLOT = [None]                        # set per pair; keeps render_pair tiny


def draw_split(rng, casts):
    """One candidate split. Returns (test, val, train) as sorted cast lists."""
    pool = list(casts)
    rng.shuffle(pool)
    test = pool[:N_TEST_CASTS]
    rest = pool[N_TEST_CASTS:]
    rng.shuffle(rest)
    n_val = int(round(len(rest) * VAL_FRACTION))
    return sorted(test), sorted(rest[:n_val]), sorted(rest[n_val:])


def covers(train_casts, bearers, nonbearers):
    """doc 54 SS6.1.1: every bearer AND every non-bearer in >= 1 TRAIN cast."""
    seen = {n for c in train_casts for n in c}
    return set(bearers) <= seen and set(nonbearers) <= seen


def build(seed=SEED):
    rng = random.Random(seed)
    humans = list(INS.HUMANS)
    assert sorted(humans) == sorted(gen.HUMANS), \
        "instrument inventory != generator inventory -- population drift"
    bearers = list(INS.CATEGORIES[CATEGORY]["members"])
    nonbearers = [n for n in humans if n not in set(bearers)]
    assert len(bearers) == 4 and len(nonbearers) == 20, \
        "doc 54 SS5.2 membership drift: |B|=%d |N|=%d" % (len(bearers), len(nonbearers))

    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())
    casts = [(b, n) for b in sorted(bearers) for n in sorted(nonbearers)]
    assert len(casts) == len(bearers) * len(nonbearers) == 80

    # ---- split, with the registered coverage acceptance rule
    draws = 0
    while True:
        draws += 1
        if draws > MAX_DRAWS:
            raise RuntimeError("coverage never satisfied in %d draws at seed %d "
                               "-- STOP, do not weaken the rule" % (MAX_DRAWS, seed))
        te_c, va_c, tr_c = draw_split(rng, casts)
        if covers(tr_c, bearers, nonbearers):
            break
    split_of = {}
    for c in te_c:
        split_of[c] = "test"
    for c in va_c:
        split_of[c] = "val"
    for c in tr_c:
        split_of[c] = "train"

    # ---- items, in canonical cast order so pair ids are stable
    items, pid = [], 0
    for cast_key in casts:
        b, n = cast_key
        split = split_of[cast_key]
        for v in verbs:
            pid += 1
            key = "p%05d" % pid
            cast = [b, n]
            rng.shuffle(cast)                      # bearer not systematically wire 0
            refs = [{"name": x, "type": "human"} for x in cast]
            VERB_SLOT[0] = v
            flag, tmpl = render_pair(rng)
            q = {"type": "QB", "surface": "هل %s %s" % (v, MARKER),
                 "v": v, "a": cast[0], "p": cast[1]}
            for suffix, agent, patient, ans in (("a", b, n, "yes"),
                                                ("b", n, b, "no")):
                items.append({
                    "id": "%s_%s_%s" % (split, key, suffix),
                    "pair_id": key, "split": split, "qtype": "QB",
                    "hop_count": 1, "pro_drop_item": False,
                    "order_dependent": True, "answer": ans,
                    "K": 2, "T": 1, "flip_idx": [0],
                    "referents": refs,
                    "events": [{"verb": v, "agent": agent,
                                "patient": patient, "pro_drop": False}],
                    "sentences": [tmpl(agent, patient)],
                    "order_flags": [flag],
                    "question": q,
                })

    counts = {s: sum(1 for i in items if i["split"] == s)
              for s in ("train", "val", "test")}

    # ---- invariants, asserted BEFORE anything is written
    assert not (set(te_c) & (set(tr_c) | set(va_c))), "test casts leak into train/val"
    assert not (set(tr_c) & set(va_c)), "train and val casts overlap"
    assert len(te_c) + len(va_c) + len(tr_c) == len(casts)
    assert covers(tr_c, bearers, nonbearers), "train coverage assertion failed"
    assert counts["test"] == 300, "n_test = %d, not the registered 300" % counts["test"]
    byp = {}
    for i in items:
        byp.setdefault(i["pair_id"], []).append(i)
    for k, g in byp.items():
        assert len(g) == 2, k
        a, b_ = g
        assert a["answer"] != b_["answer"], k
        assert [r["name"] for r in a["referents"]] == \
               [r["name"] for r in b_["referents"]], k
        assert a["question"]["surface"] == b_["question"]["surface"], k
        assert a["split"] == b_["split"], k
        assert a["order_flags"] == b_["order_flags"], k
        assert a["events"][0]["agent"] == b_["events"][0]["patient"], k
    bset = set(bearers)
    for i in items:
        assert {r["name"] for r in i["referents"]} <= set(humans), i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in bset) == 1, i["id"]
        assert (i["answer"] == "yes") == (i["events"][0]["agent"] in bset), i["id"]
    bal = {s: {a: sum(1 for i in items if i["split"] == s and i["answer"] == a)
               for a in ("yes", "no")} for s in ("train", "val", "test")}
    for s, c in bal.items():
        assert c["yes"] == c["no"], "answer imbalance in %s: %s" % (s, c)

    meta = {
        "experiment": "exp68", "rung": "SB", "design_doc": DESIGN_DOC,
        "seed": seed, "is_confirm_draw": bool(seed == CONFIRM_SEED),
        "n_items": len(items), "split_counts": counts,
        "verb_inventory": {v: {"human_patients": True} for v in verbs},
        "verb_inventory_source": VT_SRC,
        "humans": humans, "objects": [],
        "category": {"key": CATEGORY,
                     "gloss": INS.CATEGORIES[CATEGORY]["gloss"],
                     "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                                "user-confirmed 2026-08-23; imported, never retyped"),
                     "glosses": {m: INS.GLOSS[m] for m in bearers}},
        "bearers": sorted(bearers), "nonbearers": sorted(nonbearers),
        "n_casts": len(casts),
        "train_casts": [list(c) for c in tr_c],
        "val_casts": [list(c) for c in va_c],
        "test_casts": [list(c) for c in te_c],
        "split_unit": "CAST -- a (bearer, non-bearer) pairing; never within a pair",
        "heldout_is": ("UNSEEN PAIRINGS, not unseen nouns (doc 54 SS6.1.1). Every "
                       "noun's feature is in-inventory by construction, which is "
                       "what makes doc 54 SS4.3's inventory-wide PCA fit legitimate "
                       "and what makes doc 54 SS10's scope limit binding: exp68 does "
                       "NOT test novel-filler generalisation."),
        "coverage_assertion": {
            "rule": ("every bearer AND every non-bearer appears in >= 1 TRAIN cast "
                     "(doc 54 SS6.1.1)"),
            "draws_until_satisfied": draws,
            "nouns_in_train_casts": len({n for c in tr_c for n in c}),
            "satisfied": True},
        "marker_token": MARKER,
        "answer_balance": bal,
        "construction": (
            "SB (doc 46 SS5): one event, two-referent cast, exactly one category "
            "bearer; twins exchange the agent/patient roles over an IDENTICAL cast "
            "with an IDENTICAL question surface and opposite labels, so V1 is "
            "exactly 0.500 and neither content alone nor position alone can solve "
            "it. exp68's category is a REAL semantic category (non-occupational "
            "state) and is carried by the per-referent FEATURE MAP -- the raw "
            "AraVec PCA coordinates of doc 54 SS6.2 -- never by the surface."),
        "invariants_asserted": [
            "test casts are disjoint from train and val casts (unseen pairings)",
            "every bearer and every non-bearer appears in >= 1 train cast",
            "every pair: identical cast, identical wire order, identical question "
            "surface, one rendering, opposite answers",
            "exactly one bearer in every cast",
            "answer == yes iff the agent is a real bearer",
            "answer classes balanced in every split",
            "n_test == 300, matching exp60/exp61 so doc 54 SS7's 0.06 bar transfers",
        ],
    }
    meta["dataset_sha256"] = comp.canonical_sha256(items)
    return items, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="doc 54 SS6.1.1: the CONFIRM draw uses seed + 1000 = "
                         "%d, registered in the doc, never chosen after the fact"
                         % CONFIRM_SEED)
    a = ap.parse_args()
    items, meta = build(a.seed)
    json.dump({"meta": meta, "items": items},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print("[exp68] wrote %s" % a.out)
    print("  category: %s (%s) -- %s"
          % (meta["category"]["key"], meta["category"]["gloss"],
             " ".join(meta["bearers"])))
    print("  casts: %d total = %d train / %d val / %d test"
          % (meta["n_casts"], len(meta["train_casts"]),
             len(meta["val_casts"]), len(meta["test_casts"])))
    print("  splits: %s" % meta["split_counts"])
    print("  coverage satisfied after %d draw(s); %d/24 nouns in train casts"
          % (meta["coverage_assertion"]["draws_until_satisfied"],
             meta["coverage_assertion"]["nouns_in_train_casts"]))
    print("  answer balance: %s" % meta["answer_balance"])
    print("  sha256: %s" % meta["dataset_sha256"])
    ex = items[0]
    print("  example: %s | q: %s | answer %s"
          % (ex["sentences"][0], ex["question"]["surface"], ex["answer"]))
    print("  verbs: %s" % " ".join(sorted(meta["verb_inventory"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
