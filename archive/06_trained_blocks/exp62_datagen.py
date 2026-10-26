# -*- coding: utf-8 -*-
"""exp62_datagen.py -- doc 48 SS8.6: the SB rung with a THREE-WAY inventory.

Four splits, so one training run can be selected two ways (doc 48 SS8.1):
  train        train inventory, 80% of its pairs
  val_indist   train inventory, the other 20% of pairs  (exp60/61 convention)
  val_novel    a DISJOINT inventory  -> selection on NOVEL FILLERS
  test         a THIRD disjoint inventory

Construction is exp60's SB rung unchanged (doc 46 SS5); only the inventory
partition differs. Casts never cross inventories.
"""
import argparse, json, random, sys
import exp41_story_generator as gen
import exp42_compiler as comp

DESIGN_DOC = ("qnlp_private_docs/48_exp62_selection_optimisation_width.md "
              "(19e69d2, FROZEN incl. SS8 amendment)")
VT_SRC = "stories_exp51_VT.json"
OUT = "stories_exp62_SB3.json"
SEED = 62000
N_TRAIN, N_VALNOVEL, N_TEST = 10, 6, 8          # doc 48 SS8.6
VAL_INDIST_FRACTION = 0.20
MARKER = "المميز"


def build(seed=SEED):
    rng = random.Random(seed)
    humans = sorted(gen.HUMANS)
    rng.shuffle(humans)
    tr_inv = humans[:N_TRAIN]
    vn_inv = humans[N_TRAIN:N_TRAIN + N_VALNOVEL]
    te_inv = humans[N_TRAIN + N_VALNOVEL:N_TRAIN + N_VALNOVEL + N_TEST]
    halves = lambda p: (p[:len(p) // 2], p[len(p) // 2:])
    (tb, tn), (vb, vn), (eb, en) = halves(tr_inv), halves(vn_inv), halves(te_inv)
    bearers = sorted(tb + vb + eb)
    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())

    def make(split, bpool, npool, pid0):
        items, pid = [], pid0
        for b in bpool:
            for n in npool:
                for v in verbs:
                    pid += 1
                    key = "p%05d" % pid
                    cast = [b, n]
                    rng.shuffle(cast)
                    refs = [{"name": x, "type": "human"} for x in cast]
                    flag = "VSO" if rng.random() < 0.5 else "SVO"
                    tmpl = ((lambda a, p: "%s %s %s" % (v, a, p)) if flag == "VSO"
                            else (lambda a, p: "%s %s %s" % (a, v, p)))
                    q = {"type": "QB", "surface": "هل %s %s" % (v, MARKER),
                         "v": v, "a": cast[0], "p": cast[1]}
                    for suf, ag, pa, ans in (("a", b, n, "yes"), ("b", n, b, "no")):
                        items.append({
                            "id": "%s_%s_%s" % (split, key, suf), "pair_id": key,
                            "split": split, "qtype": "QB", "hop_count": 1,
                            "pro_drop_item": False, "order_dependent": True,
                            "answer": ans, "K": 2, "T": 1, "flip_idx": [0],
                            "referents": refs,
                            "events": [{"verb": v, "agent": ag, "patient": pa,
                                        "pro_drop": False}],
                            "sentences": [tmpl(ag, pa)], "order_flags": [flag],
                            "question": q})
        return items, pid

    tr_all, pid = make("train", tb, tn, 0)
    vn_items, pid = make("val_novel", vb, vn, 50000)
    te_items, _ = make("test", eb, en, 90000)

    pids = sorted({i["pair_id"] for i in tr_all})
    rng.shuffle(pids)
    vi = set(pids[:int(round(len(pids) * VAL_INDIST_FRACTION))])
    for i in tr_all:
        if i["pair_id"] in vi:
            i["split"] = "val_indist"
            i["id"] = i["id"].replace("train_", "val_indist_", 1)
    items = tr_all + vn_items + te_items

    counts = {s: sum(1 for i in items if i["split"] == s)
              for s in ("train", "val_indist", "val_novel", "test")}
    meta = {"experiment": "exp62-A", "rung": "SB3", "design_doc": DESIGN_DOC,
            "seed": seed, "n_items": len(items), "split_counts": counts,
            "verb_inventory": {v: {"human_patients": True} for v in verbs},
            "humans": humans, "objects": [],
            "train_inventory": sorted(tr_inv),
            "valnovel_inventory": sorted(vn_inv),
            "heldout_inventory": sorted(te_inv),      # name kept for exp61_bands
            "bearers": bearers, "marker_token": MARKER,
            "answer_balance": {s: {a: sum(1 for i in items if i["split"] == s
                                          and i["answer"] == a)
                                   for a in ("yes", "no")} for s in counts},
            "construction": ("exp60 SB rung with a THREE-WAY inventory (doc 48 "
                             "SS8.6). val_novel and test inventories are disjoint "
                             "from train and from each other, so a run can be "
                             "selected on NOVEL FILLERS and tested on a third, "
                             "unseen inventory."),
            "invariants_asserted": [
                "train / val_novel / test inventories mutually DISJOINT",
                "casts never cross inventories",
                "every pair: identical cast, identical question, opposite answers",
                "cast order shuffled per pair; one rendering per item"]}
    # ---- invariants asserted before anything is written
    assert len(set(tr_inv) & set(vn_inv)) == 0 and len(set(tr_inv) & set(te_inv)) == 0 \
        and len(set(vn_inv) & set(te_inv)) == 0, "inventories overlap"
    inv_of = {"train": tr_inv, "val_indist": tr_inv,
              "val_novel": vn_inv, "test": te_inv}
    byp = {}
    for i in items:
        byp.setdefault(i["pair_id"], []).append(i)
        assert {r["name"] for r in i["referents"]} <= set(inv_of[i["split"]]), i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in bearers) == 1, i["id"]
    for k, g in byp.items():
        assert len(g) == 2 and g[0]["answer"] != g[1]["answer"], k
        assert [r["name"] for r in g[0]["referents"]] == \
               [r["name"] for r in g[1]["referents"]], k
        assert g[0]["question"]["surface"] == g[1]["question"]["surface"], k
        assert g[0]["split"] == g[1]["split"], k
    meta["dataset_sha256"] = comp.canonical_sha256(items)
    return items, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    items, meta = build(a.seed)
    json.dump({"meta": meta, "items": items}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[exp62] wrote %s" % a.out)
    print("  splits: %s" % meta["split_counts"])
    print("  train %s" % meta["train_inventory"])
    print("  val_novel %s" % meta["valnovel_inventory"])
    print("  test %s" % meta["heldout_inventory"])
    print("  balance: %s" % meta["answer_balance"])
    print("  sha256: %s" % meta["dataset_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
