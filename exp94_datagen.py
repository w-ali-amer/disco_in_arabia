# -*- coding: utf-8 -*-
"""exp94_datagen.py -- doc 89 §3: agreement-paradigm twin stories for one draw (shared by all seeds; seeds vary init).

Splits (pairs): train 1600 (K in {2,3}, 5 seen classes), val 200, test_id 300, test_oodc 300 (K in {2,3}, f-du involved),
test_oodk 300 (K in {4,5}, seen classes), test_oodck 300 (K in {4,5}, f-du involved), train_full 1600 (TF-FULL: all 6
classes incl. f-du, gen seed +1).  A pair = same story, two suffixes, labels flipped.
f-du pair types (OODC/OODCK), half each: T1 X = f-du candidate (yes-suffix f-du, no-suffix another's class);
T2 X != f-du (yes-suffix X's class, no-suffix f-du).
--corrupt (in-memory; G-V designed-to-fail): the event suffix always takes the FIRST-listed candidate's class.
"""
import argparse, random
import exp94_common as C

DRAWS = {"primary": 94000, "confirm": 94500, "smoke": 94900}
SIZES = {"train": 1600, "val": 200, "test_id": 300, "test_oodc": 300, "test_oodk": 300, "test_oodck": 300,
         "train_full": 1600}
SEEN = [c for c in C.CLASSES if c != C.HELD]


def make_story(rng, K, classes_pool, must_hold=False, ptype=None):
    if must_hold:
        others = rng.sample(SEEN, K - 1)
        cls = others + [C.HELD]
        rng.shuffle(cls)
    else:
        cls = rng.sample(classes_pool, K)
    stems = rng.sample(C.STEMS, K)
    verb, obj = rng.choice(C.VERBS), rng.choice(C.OBJECTS)
    if must_hold:
        h = cls.index(C.HELD)
        if ptype == "T1":
            x = h; no_cls = cls[rng.choice([i for i in range(K) if i != h])]
        else:
            x = rng.choice([i for i in range(K) if i != h]); no_cls = C.HELD
    else:
        x = rng.randrange(K); no_cls = cls[rng.choice([i for i in range(K) if i != x])]
    return cls, stems, verb, obj, x, no_cls


def render(cls, stems, verb, obj, x, suffix_cls):
    intro = C.INTRO + [C.noun_surface(s, c, conj=(i > 0)) for i, (s, c) in enumerate(zip(stems, cls))]
    event = [C.THEN, C.verb_surface(verb, suffix_cls), C.nfc(obj + "َ")]
    # VSO question with an overt postverbal subject: MSA PARTIAL agreement -- singular verb, gender only
    # (doc 89 §2 / lit report: number is not marked on a verb preceding its overt subject). Not compiled.
    q = [C.Q, C.verb_surface(verb, (cls[x][0], "sg")), C.noun_surface(stems[x], cls[x]), C.nfc(obj + "َ"), "؟"]
    agent = cls.index(suffix_cls)
    return {"intro": intro, "event": event, "q": q, "label": int(agent == x), "gold_agent": agent,
            "meta": {"K": len(cls), "classes": ["%s-%s" % c for c in cls], "x": x,
                     "suffix": "%s-%s" % suffix_cls}}


def split(rng, n_pairs, Ks, pool, tag, must_hold=False, corrupt=False):
    out = []
    for k in range(n_pairs):
        K = Ks[k % len(Ks)]
        ptype = ("T1", "T2")[k % 2] if must_hold else None
        cls, stems, verb, obj, x, no_cls = make_story(rng, K, pool, must_hold, ptype)
        if corrupt:
            st = render(cls, stems, verb, obj, x, cls[0])
            st["id"], st["pair"] = "%s|%04d|c" % (tag, k), "%s|%04d" % (tag, k)
            out.append(st); continue
        for j, suf in enumerate((cls[x], no_cls)):
            st = render(cls, stems, verb, obj, x, suf)
            st["id"], st["pair"], st["ptype"] = "%s|%04d|%d" % (tag, k, j), "%s|%04d" % (tag, k), ptype
            out.append(st)
    return out


def build(draw, corrupt=False):
    g = DRAWS[draw]
    rng = random.Random(g + (7 if corrupt else 0))
    d = {"draw": draw, "gen_seed": g, "sizes": SIZES}
    d["train"] = split(rng, SIZES["train"], [2, 3], SEEN, "train", corrupt=corrupt)
    d["val"] = split(rng, SIZES["val"], [2, 3], SEEN, "val", corrupt=corrupt)
    d["test_id"] = split(rng, SIZES["test_id"], [2, 3], SEEN, "tid", corrupt=corrupt)
    d["test_oodc"] = split(rng, SIZES["test_oodc"], [2, 3], None, "oodc", must_hold=True, corrupt=corrupt)
    d["test_oodk"] = split(rng, SIZES["test_oodk"], [4, 5], SEEN, "oodk", corrupt=corrupt)
    d["test_oodck"] = split(rng, SIZES["test_oodck"], [4, 5], None, "oodck", must_hold=True, corrupt=corrupt)
    rf = random.Random(g + 1)
    d["train_full"] = split(rf, SIZES["train_full"], [2, 3], C.CLASSES, "full", corrupt=corrupt)
    fixed = random.Random(94000)                                  # FIXED intro angles (not trained), all draws
    d["intro_angles"] = {s: fixed.uniform(0, 3.141592653589793) for s in C.STEMS + C.OBJECTS}
    d["provenance"] = C.provenance()
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draw", required=True, choices=list(DRAWS)); a = ap.parse_args()
    d = build(a.draw)
    C.write_new_json("stories_exp94_%s.json" % a.draw, d)
    print("stories_exp94_%s.json" % a.draw, {k: len(v) for k, v in d.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
