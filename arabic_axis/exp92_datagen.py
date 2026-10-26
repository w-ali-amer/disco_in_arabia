# -*- coding: utf-8 -*-
"""exp92_datagen.py -- doc 87 §2/§5: item pools and held-out partitions for one draw.

Usage: python exp92_datagen.py --draw {primary,confirm} [--corrupt]  (--corrupt = G-V designed-to-fail draw)
Writes stories_exp92_<draw>.json (Rule 3). Per cell (pair, form):
  Form I cells:   train 40, val 12, test_seen 20
  derived cells:  train 40, val 12, test_heldout 60, test_seen 20
(val 12 not 10: build ruling doc 87 §12.1 -- 50% YES with an exact 50/50 NO split needs counts divisible by 4.)
Each item: YES = (undergoer, S_root); NO = half (non-undergoer, S_root), half (undergoer, S_other); none
(non-undergoer, S_other). S_other is drawn from states of roots in the same alternation AND undergoer class,
choosing the least-used distractor state so far (ties broken by the draw rng) to spread distractor use.
"""
import argparse, json, random, os
import exp92_common as C

DRAWS = {"primary": {"gen": 92000, "part": 92100}, "confirm": {"gen": 92500, "part": 92600}}
SIZES = {"train": 40, "val": 12, "test_heldout": 60, "test_seen": 20}
N_SEEDS = 8
N_HELD = 4


def partitions(roots, alt, rng):
    """8 blocks of 4 held-out roots; every root held out >= 2 times; no duplicate within a block."""
    names = [r["pair"] for r in roots if r["alt"] == alt]
    need = N_SEEDS * N_HELD
    for _ in range(10000):
        seq = []
        while len(seq) < need:
            p = names[:]
            rng.shuffle(p)
            seq += p
        seq = seq[:need]
        blocks = [seq[i * N_HELD:(i + 1) * N_HELD] for i in range(N_SEEDS)]
        if all(len(set(b)) == N_HELD for b in blocks) and all(seq.count(n) >= 2 for n in names):
            return blocks
    raise RuntimeError("partition constraint unsatisfiable")


def make_cell(root, form, split, n, rng, peers, dist_count, corrupt=False):
    inv = C.INAN if root["cls"] == "INAN" else C.HUMAN
    upos = C.UNDERGOER_POS[(root["alt"], form)]
    n_yes = n // 2
    n_no_a = (n - n_yes) // 2          # (non-undergoer, S_root)
    n_no_b = n - n_yes - n_no_a        # (undergoer, S_other)
    if corrupt:
        n_no_a, n_no_b = n - n_yes, 0
    kinds = ["yes"] * n_yes + ["nonU_Sr"] * n_no_a + ["U_So"] * n_no_b
    rng.shuffle(kinds)
    items = []
    for k, kind in enumerate(kinds):
        n1, n2 = rng.sample(inv, 2)
        toks = C.render(root, form, n1, n2)
        if kind == "yes":
            qpos, state = upos, root["pair"]
        elif kind == "nonU_Sr":
            qpos, state = 1 - upos, root["pair"]
        else:
            m = min(dist_count[p] for p in peers)
            cands = sorted(p for p in peers if dist_count[p] == m)
            state = rng.choice(cands)
            dist_count[state] += 1
            qpos = upos
        qnoun = (n1, n2)[qpos]
        items.append({"id": "%s|%s|%s|%03d" % (root["pair"], form, split, k), "pair": root["pair"],
                      "alt": root["alt"], "form": form, "split": split, "tokens": toks,
                      "question": ["هل", C.noun_surface(qnoun, C.NOM), C.STATE_SURFACE[state] + "؟"],
                      "q_pos": qpos, "q_state": state, "label": 1 if kind == "yes" else 0, "kind": kind,
                      "undergoer_pos": upos})
    return items


def build(draw, corrupt=False):
    cfg = DRAWS[draw]
    roots = C.load_lexicon()
    rng = random.Random(cfg["gen"] + (7 if corrupt else 0))
    items = []
    for r in roots:
        peers = [p["pair"] for p in roots if p["alt"] == r["alt"] and p["cls"] == r["cls"] and p["pair"] != r["pair"]]
        dist_count = {p: 0 for p in peers}
        for form in ("I", r["derived"]):
            splits = ["train", "val", "test_seen"] + (["test_heldout"] if form != "I" else [])
            for sp in splits:
                items += make_cell(r, form, sp, SIZES[sp], rng, peers, dist_count, corrupt)
    prng = random.Random(cfg["part"])
    held = {"ANTI": partitions(roots, "ANTI", prng), "CAUS": partitions(roots, "CAUS", prng)}
    return {"draw": draw, "corrupt": corrupt, "gen_seed": cfg["gen"], "part_seed": cfg["part"],
            "sizes": SIZES, "held_out": {str(s): held["ANTI"][s] + held["CAUS"][s] for s in range(N_SEEDS)},
            "roots": roots, "nouns": C.NOUNS, "items": items, "provenance": C.provenance()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, choices=list(DRAWS))
    ap.add_argument("--corrupt", action="store_true")
    a = ap.parse_args()
    d = build(a.draw, a.corrupt)
    out = "stories_exp92_%s%s.json" % (a.draw, "_CORRUPT" if a.corrupt else "")
    C.write_new_json(out, d)
    print(out, len(d["items"]), "items; held-out per seed:", d["held_out"]["0"])


if __name__ == "__main__":
    main()
