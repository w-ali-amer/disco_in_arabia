# -*- coding: utf-8 -*-
"""exp93_datagen.py -- doc 88 §2/§5: agreement-twin stories per seed partition, one draw.

Usage: python exp93_datagen.py --draw {primary,confirm}  -> stories_exp93_<draw>.json (Rule 3)
Per seed s (held-out block s of 4 stems): train 800 pairs, val 100, test_heldout 200 (both S1 humans from held-out
stems), test_seen 200 (training stems). A pair = two stories identical except V2's suffix; labels flipped.
Balance: the 4 combos (S1-subject gender) x (X = H_a / H_b) cycle exactly within every split.
--corrupt (in-memory, for G-V designed-to-fail): no twins; V2 always agrees with the S1 subject.
"""
import argparse, random
import exp93_common as C

DRAWS = {"primary": {"gen": 93000, "part": 93100}, "confirm": {"gen": 93500, "part": 93600},
         "smoke": {"gen": 93900, "part": 93950}}
SIZES = {"train": 800, "val": 100, "test_heldout": 200, "test_seen": 200}
N_SEEDS, N_HELD = 8, 4


def partitions(rng):
    names = list(C.STEMS)
    for _ in range(10000):
        seq = []
        for _ in range(2):
            p = names[:]; rng.shuffle(p); seq += p
        blocks = [seq[i * N_HELD:(i + 1) * N_HELD] for i in range(N_SEEDS)]
        if all(len(set(b)) == N_HELD for b in blocks):
            return blocks
    raise RuntimeError


def story(m_stem, f_stem, subj_fem, x_is_a, v1, v2, obj, suffix_fem):
    a = (f_stem, 1) if subj_fem else (m_stem, 0)            # H_a = S1 subject
    b = (m_stem, 0) if subj_fem else (f_stem, 1)
    x = a if x_is_a else b
    s1 = [C.verb_surface(C.V1[v1], a[1]), C.noun_surface(a[0], a[1], C.NOM), C.noun_surface(b[0], b[1], C.ACC)]
    s2 = [C.THEN, C.verb_surface(C.V2[v2], suffix_fem), C.noun_surface(obj, 0, C.ACC)]
    q = [C.Q, C.verb_surface(C.V2[v2], x[1]), C.noun_surface(x[0], x[1], C.NOM), C.noun_surface(obj, 0, C.ACC), "؟"]
    agent_is_a = (a[1] == suffix_fem)
    label = int(x[1] == suffix_fem)
    return {"s1": s1, "s2": s2, "q": q, "label": label, "gold_agent_wire": 0 if agent_is_a else 1,
            "meta": {"subj_fem": subj_fem, "x_is_a": x_is_a, "x_fem": x[1], "suffix_fem": suffix_fem}}


def make_split(stems, n_pairs, rng, tag, corrupt=False):
    out = []
    combos = [(sf, xa) for sf in (0, 1) for xa in (1, 0)]
    for k in range(n_pairs):
        sf, xa = combos[k % 4]
        m, f = rng.sample(stems, 2)                            # different stems for the two genders
        v1, v2, obj = rng.choice(sorted(C.V1)), rng.choice(sorted(C.V2)), rng.choice(C.OBJECTS)
        if corrupt:
            st = story(m, f, sf, xa, v1, v2, obj, suffix_fem=sf)   # V2 always agrees with S1 subject
            st["id"], st["pair"] = "%s|%04d|c" % (tag, k), "%s|%04d" % (tag, k)
            out.append(st)
            continue
        for suf in (0, 1):
            st = story(m, f, sf, xa, v1, v2, obj, suf)
            st["id"], st["pair"] = "%s|%04d|%d" % (tag, k, suf), "%s|%04d" % (tag, k)
            out.append(st)
    return out


def build(draw, corrupt=False):
    cfg = DRAWS[draw]
    blocks = partitions(random.Random(cfg["part"]))
    rng = random.Random(cfg["gen"] + (7 if corrupt else 0))
    fixed = random.Random(93000)                                 # FIXED intro angles, same for all draws/arms
    angles = {s: fixed.uniform(0, 3.141592653589793) for s in C.STEMS + C.OBJECTS}
    seeds = {}
    for s in range(N_SEEDS):
        held = blocks[s]; train_stems = [x for x in C.STEMS if x not in held]
        seeds[str(s)] = {"held_out": held,
                         "train": make_split(train_stems, SIZES["train"], rng, "s%d_train" % s, corrupt),
                         "val": make_split(train_stems, SIZES["val"], rng, "s%d_val" % s, corrupt),
                         "test_heldout": make_split(held, SIZES["test_heldout"], rng, "s%d_th" % s, corrupt),
                         "test_seen": make_split(train_stems, SIZES["test_seen"], rng, "s%d_ts" % s, corrupt)}
    return {"draw": draw, "corrupt": corrupt, "cfg": cfg, "sizes": SIZES, "intro_angles": angles, "seeds": seeds,
            "provenance": C.provenance()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, choices=list(DRAWS))
    a = ap.parse_args()
    d = build(a.draw)
    C.write_new_json("stories_exp93_%s.json" % a.draw, d)
    print("stories_exp93_%s.json" % a.draw, sum(len(v) for sd in d["seeds"].values() for k, v in sd.items() if k != "held_out"))


if __name__ == "__main__":
    main()
