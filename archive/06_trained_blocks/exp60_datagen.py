# -*- coding: utf-8 -*-
"""exp60_datagen.py -- doc 46 Stage A: the SEMANTIC BINDING (SB) rung.

doc 46 SS1: every rung this programme built makes referent identity provably
irrelevant -- both twins share a cast, so any function of the nouns alone is
constant within a pair and scores exactly 0.500. SB is the first rung where
meaning is LOAD-BEARING.

CONSTRUCTION (doc 46 SS5). One event, a two-referent cast, exactly one
CATEGORY-BEARER. The question asks whether the bearer filled the AGENT role.

    twin A   event (v, agent=bearer,     patient=non-bearer)   -> yes
    twin B   event (v, agent=non-bearer, patient=bearer)       -> no

Identical cast, identical question surface, opposite labels. So:
  * V1 bag-of-embeddings is EXACTLY 0.500 by construction;
  * position alone cannot solve it -- you must know WHICH referent is the bearer;
  * content alone cannot solve it -- the bag is identical;
  * the answer requires BINDING content to role.

THE CATEGORY IS NOT IN THE SURFACE. It is carried only by the angle map built at
training time from (Delta, w) -- so a token-reading model can memorise
noun->bearer on the TRAIN inventory but cannot transfer, while an angle-reading
model transfers by construction. That asymmetry is the whole instrument, and it
is why the split is by REFERENT INVENTORY (doc 46 SS5): train/val use 14 nouns,
test uses a disjoint 10. Novel fillers are the strongest available measure of
role-filler binding (doc 46 SS2.2).

Registered details:
  * cast order is SHUFFLED per pair, so the bearer is not systematically wire 0;
  * both twins of a pair share ONE word-order rendering, so the only surface
    difference is the noun swap;
  * each item is rendered ONCE -- emitting both VSO and SVO would produce
    CIRCUIT-LEVEL duplicates, since the compiled circuit ignores surface order
    (doc 46 SS5, V4);
  * the question surface is constant modulo the verb and names no referent.
"""
import argparse, hashlib, json, random, sys

import exp41_story_generator as gen
import exp42_compiler as comp

DESIGN_DOC = ("qnlp_private_docs/46_exp60_semantic_binding_and_reuploading.md "
              "(a45a2e9, FROZEN)")
VT_SRC = "stories_exp51_VT.json"          # verb inventory provenance
OUT = "stories_exp60_SB.json"
SEED = 60000
N_TRAIN_NOUNS, N_HELD_NOUNS = 14, 10      # doc 46 SS5, frozen
VAL_FRACTION = 0.20
MARKER = "المميز"                          # placeholder; names no referent


def render_pair(rng, verb, agent, patient):
    """One rendering per item; both twins share the order flag."""
    if rng.random() < 0.5:
        return "VSO", (lambda a, p: "%s %s %s" % (verb, a, p))
    return "SVO", (lambda a, p: "%s %s %s" % (a, verb, p))


def build(seed=SEED):
    rng = random.Random(seed)
    humans = sorted(gen.HUMANS)
    rng.shuffle(humans)
    train_nouns, held_nouns = humans[:N_TRAIN_NOUNS], \
        humans[N_TRAIN_NOUNS:N_TRAIN_NOUNS + N_HELD_NOUNS]
    # bearers are HALF of each inventory, so the category is balanced in both
    tb, tn = train_nouns[:N_TRAIN_NOUNS // 2], train_nouns[N_TRAIN_NOUNS // 2:]
    hb, hn = held_nouns[:N_HELD_NOUNS // 2], held_nouns[N_HELD_NOUNS // 2:]
    bearers = sorted(tb + hb)

    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())

    def make(split, bearer_pool, non_pool, pid_start):
        items, pid = [], pid_start
        for b in bearer_pool:
            for n in non_pool:
                for v in verbs:
                    pid += 1
                    key = "p%05d" % pid
                    cast = [b, n]
                    rng.shuffle(cast)                       # bearer not always wire 0
                    refs = [{"name": x, "type": "human"} for x in cast]
                    flag, tmpl = render_pair(rng, v, None, None)
                    q = {"type": "QB", "surface": "هل %s %s" % (v, MARKER),
                         "v": v, "a": cast[0], "p": cast[1]}
                    for suffix, agent, patient, ans in (
                            ("a", b, n, "yes"), ("b", n, b, "no")):
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
        return items, pid

    tr_all, pid = make("train", tb, tn, 0)
    te, _ = make("test", hb, hn, 90000)

    # val is a disjoint PAIR subset of the train inventory -- in-distribution
    pids = sorted({i["pair_id"] for i in tr_all})
    rng.shuffle(pids)
    n_val = int(round(len(pids) * VAL_FRACTION))
    val_pids = set(pids[:n_val])
    for i in tr_all:
        if i["pair_id"] in val_pids:
            i["split"] = "val"
            i["id"] = i["id"].replace("train_", "val_", 1)
    items = tr_all + te

    counts = {s: sum(1 for i in items if i["split"] == s)
              for s in ("train", "val", "test")}
    meta = {
        "experiment": "exp60", "rung": "SB", "design_doc": DESIGN_DOC,
        "seed": seed, "n_items": len(items), "split_counts": counts,
        "verb_inventory": {v: {"human_patients": True} for v in verbs},
        "humans": humans, "objects": [],
        "train_inventory": sorted(train_nouns),
        "heldout_inventory": sorted(held_nouns),
        "bearers": bearers,
        "train_bearers": sorted(tb), "heldout_bearers": sorted(hb),
        "marker_token": MARKER,
        "answer_balance": {s: {a: sum(1 for i in items if i["split"] == s
                                      and i["answer"] == a)
                               for a in ("yes", "no")}
                           for s in ("train", "val", "test")},
        "construction": ("SB: one event, two-referent cast, exactly one "
                         "category-bearer; twins exchange the agent/patient "
                         "roles; question asks whether the bearer was the "
                         "agent. Cast and question identical across twins, "
                         "labels opposite -> V1 exactly 0.500. Category is "
                         "carried ONLY by the angle map (built at train time "
                         "from Delta,w), never by the surface."),
        "invariants_asserted": [
            "train/val and test referent inventories are DISJOINT",
            "every pair: identical cast, identical question surface, "
            "opposite answers",
            "cast order shuffled per pair",
            "one rendering per item (no circuit-level duplicates)",
        ],
    }
    # ---- invariants, asserted before anything is written
    assert not (set(train_nouns) & set(held_nouns)), "inventories overlap"
    byp = {}
    for i in items:
        byp.setdefault(i["pair_id"], []).append(i)
    for k, g in byp.items():
        assert len(g) == 2, k
        a, b = g
        assert a["answer"] != b["answer"], k
        assert [r["name"] for r in a["referents"]] == \
               [r["name"] for r in b["referents"]], k
        assert a["question"]["surface"] == b["question"]["surface"], k
        assert a["split"] == b["split"], k
    for i in items:
        pool = set(train_nouns) if i["split"] in ("train", "val") \
            else set(held_nouns)
        assert {r["name"] for r in i["referents"]} <= pool, i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in bearers) == 1, i["id"]
    meta["dataset_sha256"] = comp.canonical_sha256(items)
    return items, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="doc 46 SS10.4: the CONFIRM draw uses "
                         "dataset_seed + 1000 = 61000, registered "
                         "in the doc, never chosen after the fact")
    args = ap.parse_args()
    items, meta = build(args.seed)
    json.dump({"meta": meta, "items": items}, open(args.out, "w",
              encoding="utf-8"), ensure_ascii=False)
    print("[exp60] wrote %s" % args.out)
    print("  splits: %s" % meta["split_counts"])
    print("  train inventory %d, held-out %d, disjoint"
          % (len(meta["train_inventory"]), len(meta["heldout_inventory"])))
    print("  bearers: %d of %d nouns" % (len(meta["bearers"]),
                                         len(meta["humans"])))
    print("  answer balance: %s" % meta["answer_balance"])
    print("  sha256: %s" % meta["dataset_sha256"])
    ex = items[0]
    print("  example: %s | q: %s | answer %s"
          % (ex["sentences"][0], ex["question"]["surface"], ex["answer"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
