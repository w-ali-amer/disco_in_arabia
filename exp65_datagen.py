# -*- coding: utf-8 -*-
"""exp65_datagen.py -- doc 51: the WAZN-BINDING rung (فاعِل vs non-فاعِل).

SB's construction (doc 46 SS5) with the constructed 1-bit category replaced by a
REAL morphological contrast. Twins share cast and question with opposite labels,
so V1 stays exactly 0.500; the answer depends on WHICH class filled the AGENT
role, so the class is load-bearing only through binding.

Split is by INVENTORY (doc 51 SS11.6): train/val nouns disjoint from test nouns.
With one word per root this is WORD-holdout, not root-holdout (SS11.8).
"""
import argparse, json, random, sys
import exp41_story_generator as gen
import exp42_compiler as comp
import exp65_featurizer as F

DESIGN_DOC = ("qnlp_private_docs/51_exp65_wazn_binding_PROPOSAL.md "
              "(d293f42, FROZEN incl. SS11-12)")
VT_SRC = "stories_exp51_VT.json"
OUT = "stories_exp65_WZ.json"
SEED = 65100
VAL_FRACTION = 0.20
MARKER = "المميز"


def build(seed=SEED):
    rng = random.Random(seed)
    tr_inv, te_inv = F.split_inventory()
    tb = [n for n in tr_inv if n in F.FAAIL]; tn = [n for n in tr_inv if n not in F.FAAIL]
    eb = [n for n in te_inv if n in F.FAAIL]; en = [n for n in te_inv if n not in F.FAAIL]
    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())

    def make(split, bpool, npool, pid0):
        items, pid = [], pid0
        for b in bpool:
            for n in npool:
                for v in verbs:
                    pid += 1
                    key = "p%05d" % pid
                    cast = [b, n]; rng.shuffle(cast)
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
    te_items, _ = make("test", eb, en, 90000)
    pids = sorted({i["pair_id"] for i in tr_all})
    rng.shuffle(pids)
    vi = set(pids[:int(round(len(pids) * VAL_FRACTION))])
    for i in tr_all:
        if i["pair_id"] in vi:
            i["split"] = "val"; i["id"] = i["id"].replace("train_", "val_", 1)
    items = tr_all + te_items
    counts = {s: sum(1 for i in items if i["split"] == s)
              for s in ("train", "val", "test")}
    meta = {"experiment": "exp65", "rung": "WZ", "design_doc": DESIGN_DOC,
            "seed": seed, "split_seed": F.SPLIT_SEED, "n_items": len(items),
            "split_counts": counts,
            "verb_inventory": {v: {"human_patients": True} for v in verbs},
            "humans": sorted(gen.HUMANS), "objects": [],
            "train_inventory": sorted(tr_inv), "heldout_inventory": sorted(te_inv),
            "bearers": sorted(F.FAAIL), "contrast": "فاعِل vs non-فاعِل",
            "marker_token": MARKER,
            "answer_balance": {s: {a: sum(1 for i in items if i["split"] == s
                                          and i["answer"] == a) for a in ("yes", "no")}
                               for s in counts},
            "construction": ("SB rung with the category = a REAL morphological "
                             "contrast. Class carried ONLY by the angle map, "
                             "computed from surface form by a train-fitted "
                             "positional featurizer (doc 51 SS4/SS11.1)."),
            "invariants_asserted": [
                "train/val and test inventories DISJOINT",
                "every pair: identical cast + question, opposite answers",
                "exactly one فاعِل referent per cast",
                "cast order shuffled per pair; one rendering per item"]}
    assert not (set(tr_inv) & set(te_inv)), "inventories overlap"
    byp = {}
    for i in items:
        byp.setdefault(i["pair_id"], []).append(i)
        pool = set(tr_inv) if i["split"] in ("train", "val") else set(te_inv)
        assert {r["name"] for r in i["referents"]} <= pool, i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in F.FAAIL) == 1, i["id"]
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
    a = ap.parse_args()
    items, meta = build()
    json.dump({"meta": meta, "items": items}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[exp65] wrote %s" % a.out)
    print("  splits: %s" % meta["split_counts"])
    print("  train inv %d (faail %d) | test inv %d (faail %d)"
          % (len(meta["train_inventory"]),
             len(set(meta["train_inventory"]) & F.FAAIL),
             len(meta["heldout_inventory"]),
             len(set(meta["heldout_inventory"]) & F.FAAIL)))
    print("  balance: %s" % meta["answer_balance"])
    print("  sha256: %s" % meta["dataset_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
