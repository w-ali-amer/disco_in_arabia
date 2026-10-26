# -*- coding: utf-8 -*-
"""exp90_datagen.py -- doc 83 SS3 / SS7 / SS11.1: render Arabic CASE-TWINS
from an exp60 SB draw.

Item types (SS3.2):
  CONSISTENT  agent in slot 1 (NOM first). Surface order = the exp60 item's
              own VSO/SVO flag (VSO == SVO in topology).
  CONFLICT    patient in slot 1 (ACC first). Rendered VOS (SS2 consequence:
              OVS is essentially unattested; an OVS string is stored in
              `surface_ovs` for the descriptive leg only).
Vocalisation (SS3.1): agent -u (U+064F), patient -a (U+064E); nisba nouns
-iyyu / -iyya (shadda + vowel); the verb carries a final fatha; all nouns
definite, no tanwin. Surfaces are stored NFC.

TEST (twin construction rule, SS3.2, makes T1 exact): for every exp60 TEST
item X (agent a, patient p) we emit
  X__CONS : slots (a, p), cases (NOM, ACC), label = X's label
  X__CONF : slots (a, p) SAME noun order, vowels SWAPPED (ACC, NOM) -> roles
            swap (agent := p), label flips. twin_of = X__CONS.
so a vowel-blind learner sees bit-identical circuits with opposite labels.
Plus the unvocalised version of every test item (`surface_unvoc`).

TRAIN/VAL (SS7 Rung A / Rung B): each exp60 train/val EVENT is rendered once,
CONFLICT iff u_e < k, where u_e ~ U(0,1) is drawn ONCE per event (events in
sorted-id order) from random.Random(render_seed). This is an independent
Bernoulli(k) per event for every k, the doses are nested (common random
numbers), and no event can appear in both renderings. A CONFLICT training
item keeps the event (agent, patient, label) and puts the patient first:
slots (p, a), cases (ACC, NOM), VOS.

render_seed is REGISTERED here (doc 83 SS7 says "a registered seed" without a
number; build-agent choice, flagged): see RENDER_SEEDS.
"""
import argparse
import json
import os
import random
import sys

import exp90_encoder as enc
from exp90_common import (K_CONDS, DESIGN_DOC, canonical_sha256, file_sha256,
                          write_new_json)

RENDER_SEEDS = {                       # registered by the build (flagged)
    "stories_exp60_SB.json": 90010,            # primary (seed 60000)
    "stories_exp90_confirm_base.json": 90510,  # confirm (seed 90500)
    "stories_exp60_SB_confirm.json": 90610,    # exp60 CONF draw (61000), A0 only
}
OUT_FOR = {
    "stories_exp60_SB.json": "stories_exp90_primary.json",
    "stories_exp90_confirm_base.json": "stories_exp90_confirm.json",
    "stories_exp60_SB_confirm.json": "stories_exp90_A0_61000.json",
}


def render_surface(verb, slots, cases, order):
    v = enc.vocalise_verb(verb)
    n1 = enc.vocalise_noun(slots[0], cases[0])
    n2 = enc.vocalise_noun(slots[1], cases[1])
    if order in ("VSO", "VOS"):
        return "%s %s %s" % (v, n1, n2)
    if order in ("SVO", "OVS"):
        return "%s %s %s" % (n1, v, n2)
    raise ValueError(order)


def make_item(base, item_type, split, agent, patient, bearers, suffix,
              twin_of=None):
    verb = base["events"][0]["verb"]
    if item_type == "CONSISTENT":
        slots, cases, order = [agent, patient], [enc.NOM, enc.ACC], \
            base["order_flags"][0]
        assert order in ("VSO", "SVO")
    else:
        slots, cases, order = [patient, agent], [enc.ACC, enc.NOM], "VOS"
    surf = render_surface(verb, slots, cases, order)
    it = {
        "id": "%s__%s" % (base["id"], suffix), "base_id": base["id"],
        "pair_id": base["pair_id"], "split": split, "item_type": item_type,
        "order": order, "referents": [r["name"] for r in base["referents"]],
        "verb": verb, "agent": agent, "patient": patient,
        "slots": slots, "slot_cases": cases,
        "surface": surf, "surface_unvoc": enc.strip_marks(surf),
        "answer": "yes" if agent in bearers else "no",
        "question": base["question"],
    }
    if item_type == "CONFLICT":
        it["surface_ovs"] = render_surface(verb, slots, cases, "OVS")
    if twin_of:
        it["twin_of"] = twin_of
    return it


def build(base_path, render_seed):
    blob = json.load(open(base_path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("base provenance FAILED: %s" % base_path)
    bearers = set(meta["bearers"])
    for it in items:                      # exp60 label rule, re-asserted
        ev = it["events"][0]
        assert it["answer"] == ("yes" if ev["agent"] in bearers else "no")
        assert len(it["events"]) == 1 and it["K"] == 2

    # ---------------- test: CONSISTENT + case-twin CONFLICT
    test = []
    for b in sorted([i for i in items if i["split"] == "test"],
                    key=lambda x: x["id"]):
        a, p = b["events"][0]["agent"], b["events"][0]["patient"]
        cons = make_item(b, "CONSISTENT", "test", a, p, bearers, "CONS")
        # twin: same noun order (a, p), vowels swapped -> agent p, patient a
        conf = make_item(b, "CONFLICT", "test", p, a, bearers, "CONF",
                         twin_of=cons["id"])
        assert conf["slots"] == cons["slots"], b["id"]
        assert conf["slot_cases"] == cons["slot_cases"][::-1]
        assert conf["answer"] != cons["answer"], b["id"]
        assert cons["answer"] == b["answer"]
        test += [cons, conf]

    # ---------------- train/val: one uniform per event, nested doses
    rng = random.Random(render_seed)
    tv = sorted([i for i in items if i["split"] in ("train", "val")],
                key=lambda x: x["id"])
    u = {b["id"]: rng.random() for b in tv}
    train_sets, counts = {}, {}
    for cond, k in K_CONDS.items():
        out = []
        for b in tv:
            a, p = b["events"][0]["agent"], b["events"][0]["patient"]
            conflict = u[b["id"]] < k
            it = make_item(b, "CONFLICT" if conflict else "CONSISTENT",
                           b["split"], a, p, bearers,
                           "CONF" if conflict else "CONS")
            assert it["answer"] == b["answer"]
            out.append(it)
        train_sets[cond] = out
        counts[cond] = {
            "k": k,
            "n_conflict_train": sum(1 for i in out if i["split"] == "train"
                                    and i["item_type"] == "CONFLICT"),
            "n_train": sum(1 for i in out if i["split"] == "train"),
            "n_conflict_val": sum(1 for i in out if i["split"] == "val"
                                  and i["item_type"] == "CONFLICT"),
            "n_val": sum(1 for i in out if i["split"] == "val"),
        }
    assert counts["k0"]["n_conflict_train"] == 0
    m = {
        "experiment": "exp90", "design_doc": DESIGN_DOC,
        "base_file": os.path.basename(base_path),
        "base_file_sha256": file_sha256(base_path),
        "base_dataset_sha256": meta["dataset_sha256"],
        "base_seed": meta["seed"], "render_seed": render_seed,
        "render_rule": ("u_e ~ U(0,1) once per train/val event in sorted-id "
                        "order from random.Random(render_seed); CONFLICT iff "
                        "u_e < k (nested doses)"),
        "k_conds": K_CONDS, "counts": counts,
        "bearers": meta["bearers"], "humans": meta["humans"],
        "train_inventory": meta["train_inventory"],
        "heldout_inventory": meta["heldout_inventory"],
        "verbs": sorted(meta["verb_inventory"].keys()),
        "n_test": len(test),
    }
    data = {"test": test, "train_sets": train_sets}
    m["data_sha256"] = canonical_sha256(data)
    return {"meta": m, **data}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, choices=sorted(RENDER_SEEDS))
    args = ap.parse_args()
    out = OUT_FOR[args.base]
    obj = build(args.base, RENDER_SEEDS[args.base])
    sha = write_new_json(out, obj)
    print("[exp90_datagen] wrote %s  payload_sha256=%s" % (out, sha))
    print("  render_seed=%d n_test=%d" % (obj["meta"]["render_seed"],
                                           obj["meta"]["n_test"]))
    for c, v in obj["meta"]["counts"].items():
        print("  %-5s k=%.3f  n_conflict_train=%d/%d  n_conflict_val=%d/%d"
              % (c, v["k"], v["n_conflict_train"], v["n_train"],
                 v["n_conflict_val"], v["n_val"]))
    t = obj["test"][:2]
    for i in t:
        print("  %s | %s | unvoc %s | %s" % (i["item_type"], i["surface"],
                                             i["surface_unvoc"], i["answer"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
