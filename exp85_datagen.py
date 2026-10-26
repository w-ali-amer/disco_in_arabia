# -*- coding: utf-8 -*-
"""exp85_datagen.py -- doc 77 §2: Q69 generator extended to T=3.

REGISTRATION: qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)

WHAT EXP85 IS

    The 6-verb exp51 inventory (carried intact from exp69) is partitioned
    2/2/2 into THREE disjoint slots at generator seed 85999.  Event i in a
    T-event story draws its verb from slot i, so the query is resolved by
    VERB IDENTITY exactly as doc 55 §4.2 requires.  Disjointness at three
    slots carries the same cost as at two: the construction does NOT test
    discourse-position resolution.

    T=1: event 1 draws from V1 (2 verbs).
    T=2: event 1 from V1, event 2 from V2 (4 verb pairs).
    T=3: event 1 from V1, event 2 from V2, event 3 from V3 (8 verb triples).

    All T share ONE cast split (80 casts, 25 test / 11 val / 44 train), so
    the doc 55 §6.10(b) qualitative control transfers to T=3.

    Cast budget, split sizes: identical to exp69 (80 casts, 6 pairs per cast,
    2 twins, n_test=300 per T).  The 2/2/2 slot structure means:
      • T=2 has 4 verb-pair combinations (not 9);
      • T=1 has 2 verbs cycling over 6 pairs per cast (3 pairs each).

    No predictions file is required for exp85 (doc 77 registered no
    exp85-predictions step; the ordering rule is satisfied because the
    registered doc 77 precedes any code).

Usage:
    python exp85_datagen.py            # generate stories_exp85_QA.json
    python exp85_datagen.py --validate # basic self-check (sha + slot assertions)
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

import exp41_story_generator as gen
import exp42_compiler as comp
import instrument_sb_bounds as INS

SPEC = "qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)"
VT_SRC = "stories_exp51_VT.json"
OUT = "stories_exp85_QA.json"

SEED = 85999                   # registered generator seed (doc 77 §0)
N_CASTS = 80                   # same as exp69
N_TEST_CASTS = 25              # -> 150 pairs -> 300 items per T
VAL_FRACTION = 0.20            # by cast
PAIRS_PER_CAST = 6             # same as exp69 -> 960 items per T
MARKER = "المميز"
MAX_DRAWS = 500
N_TEST_ITEMS = 300             # n_test=300 per T


# ================================================================ surfaces
def render(order, verb, agent, patient):
    """SVO and VSO -- the two orders the archived pregroup reader implements."""
    if order == "SVO":
        return "%s %s %s" % (agent, verb, patient)
    return "%s %s %s" % (verb, agent, patient)


def question_surface(vq):
    return "الذي %s هل هو %s" % (vq, MARKER)


# ==================================================================== split
def draw_cast_pool(rng, bearers, nonbearers):
    """All (b, n) pairings; exactly N_CASTS=80 -> identity draw (4x20=80)."""
    allc = [(b, n) for b in bearers for n in nonbearers]
    assert len(allc) == N_CASTS, (
        "exp85 uses Part A (4 bearers × 20 non-bearers = 80 casts exactly); "
        "got %d pairings" % len(allc))
    return sorted(allc), 0


def draw_split(rng, casts):
    pool = list(casts)
    rng.shuffle(pool)
    test = pool[:N_TEST_CASTS]
    rest = pool[N_TEST_CASTS:]
    rng.shuffle(rest)
    n_val = int(round(len(rest) * VAL_FRACTION))
    return sorted(test), sorted(rest[:n_val]), sorted(rest[n_val:])


def covers(train_casts, bearers, nonbearers):
    seen = {n for c in train_casts for n in c}
    return set(bearers) <= seen and set(nonbearers) <= seen


# ==================================================================== build
def build(seed=SEED):
    """Build T=1, T=2, T=3 items under the 2/2/2 slot structure."""
    # -- Part A only (nonoccupational category, 4 bearers, 20 nonbearers)
    bearers = sorted(INS.CATEGORIES["nonoccupational"]["members"])
    nonbearers = sorted(n for n in INS.HUMANS if n not in set(bearers))
    assert (len(bearers), len(nonbearers)) == (4, 20), "Part A membership drift"
    bset = set(bearers)

    rng = random.Random(seed)

    # -- Verb inventory
    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())
    assert len(verbs) == 6, "exp51 inventory must have 6 verbs, got %d" % len(verbs)

    # -- 2/2/2 slot partition at seed 85999 (doc 77 §0, §2)
    vshuf = list(verbs)
    rng.shuffle(vshuf)
    V1, V2, V3 = (sorted(vshuf[0:2]),
                  sorted(vshuf[2:4]),
                  sorted(vshuf[4:6]))
    assert not (set(V1) & set(V2)), "V1 and V2 must be DISJOINT"
    assert not (set(V1) & set(V3)), "V1 and V3 must be DISJOINT"
    assert not (set(V2) & set(V3)), "V2 and V3 must be DISJOINT"
    assert set(V1) | set(V2) | set(V3) == set(verbs), "V1∪V2∪V3 must equal the full inventory"

    slot_assignment = {"V1": V1, "V2": V2, "V3": V3}

    # -- Cast pool (Part A: exactly 80 casts, no draw needed)
    casts, pool_draws = draw_cast_pool(rng, bearers, nonbearers)
    assert len(casts) == N_CASTS

    # -- Split with coverage requirement
    draws = 0
    while True:
        draws += 1
        if draws > MAX_DRAWS:
            raise RuntimeError(
                "train coverage never satisfied in %d draws at seed %d"
                % (MAX_DRAWS, seed))
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

    # Order repertoires
    ORDERS2 = [("SVO", "SVO"), ("SVO", "VSO"), ("VSO", "SVO"), ("VSO", "VSO")]
    ORDERS3 = [(o1, o2, o3)
               for o1 in ("SVO", "VSO")
               for o2 in ("SVO", "VSO")
               for o3 in ("SVO", "VSO")]  # 8 triples

    # Verb combination tables
    VPAIRS  = [(a, b) for a in V1 for b in V2]      # 4 pairs for T=2
    VTRIPLES = [(a, b, c) for a in V1 for b in V2 for c in V3]  # 8 for T=3

    items = []
    pid1 = pid2 = pid3 = 0

    for ci, cast_key in enumerate(casts):
        b, n = cast_key
        split = split_of[cast_key]

        # ---- T=1: 6 pairs per cast; 2 verbs from V1 cycling (3 each) -------
        for j in range(PAIRS_PER_CAST):
            pid1 += 1
            key = "p1_%05d" % pid1
            v = V1[j % 2]                          # alternating, 3 each → balanced
            order = ["SVO", "VSO"][(ci + j) % 2]
            cast = [b, n]
            rng.shuffle(cast)
            refs = [{"name": x, "type": "human"} for x in cast]
            q = {"type": "Q69", "surface": question_surface(v),
                 "v": v, "a": cast[0], "p": cast[1]}
            for suffix, agent, patient, ans in (("a", b, n, "yes"),
                                                ("b", n, b, "no")):
                items.append({
                    "id": "%s_%s_%s" % (split, key, suffix),
                    "pair_id": key, "split": split, "qtype": "Q69",
                    "hop_count": 1, "pro_drop_item": False,
                    "order_dependent": True, "answer": ans,
                    "K": 2, "T": 1, "flip_idx": [0],
                    "query_index": 1, "queried_verb": v,
                    "verb_slots": {"V1": v, "V2": None, "V3": None},
                    "referents": refs,
                    "events": [{"verb": v, "agent": agent, "patient": patient,
                                "pro_drop": False}],
                    "sentences": [render(order, v, agent, patient)],
                    "order_flags": [order],
                    "question": q,
                })

        # ---- T=2: 6 pairs per cast; 4 verb-pair combos; qpat [1,1,1,2,2,2] -
        qpat2 = [1, 1, 1, 2, 2, 2]
        rng.shuffle(qpat2)
        for j in range(PAIRS_PER_CAST):
            pid2 += 1
            key = "p2_%05d" % pid2
            va, vb = VPAIRS[(5 * ci + j) % 4]      # stride 5, coprime to 4
            o1, o2 = ORDERS2[(3 * ci + j) % 4]     # stride 3, coprime to 4
            uq_bearer_agent = ((ci + j) % 2 == 0)
            qi = qpat2[j]
            vq = va if qi == 1 else vb
            cast = [b, n]
            rng.shuffle(cast)
            refs = [{"name": x, "type": "human"} for x in cast]
            uq_agent, uq_patient = (b, n) if uq_bearer_agent else (n, b)
            qsurf = question_surface(vq)
            qspec = {"type": "Q69", "surface": qsurf, "v": vq,
                     "a": cast[0], "p": cast[1]}
            for suffix, qa, qp, ans in (("a", b, n, "yes"), ("b", n, b, "no")):
                ev = [None, None]
                ev[qi - 1] = {"verb": (va if qi == 1 else vb),
                              "agent": qa, "patient": qp, "pro_drop": False}
                ev[2 - qi] = {"verb": (vb if qi == 1 else va),
                              "agent": uq_agent, "patient": uq_patient,
                              "pro_drop": False}
                items.append({
                    "id": "%s_%s_%s" % (split, key, suffix),
                    "pair_id": key, "split": split, "qtype": "Q69",
                    "hop_count": 1, "pro_drop_item": False,
                    "order_dependent": True, "answer": ans,
                    "K": 2, "T": 2, "flip_idx": [qi - 1],
                    "query_index": qi, "queried_verb": vq,
                    "verb_slots": {"V1": va, "V2": vb, "V3": None},
                    "unqueried_bearer_is_agent": bool(uq_bearer_agent),
                    "referents": refs,
                    "events": ev,
                    "sentences": [render([o1, o2][k],
                                         ev[k]["verb"], ev[k]["agent"], ev[k]["patient"])
                                  for k in (0, 1)],
                    "order_flags": [o1, o2],
                    "question": qspec,
                })

        # ---- T=3: 6 pairs per cast; 8 verb-triple combos; qpat [1,1,2,2,3,3] -
        qpat3 = [1, 1, 2, 2, 3, 3]
        rng.shuffle(qpat3)
        for j in range(PAIRS_PER_CAST):
            pid3 += 1
            key = "p3_%05d" % pid3
            va, vb, vc = VTRIPLES[(5 * ci + j) % 8]  # stride 5, coprime to 8
            o1, o2, o3 = ORDERS3[(3 * ci + j) % 8]   # stride 3, coprime to 8
            qi = qpat3[j]
            slot_verbs = [va, vb, vc]
            vq = slot_verbs[qi - 1]
            # Unqueried events: same bearer-is-agent flag (simple, balanced)
            uq_bearer_agent = ((ci + j) % 2 == 0)
            uq_agent, uq_patient = (b, n) if uq_bearer_agent else (n, b)
            cast = [b, n]
            rng.shuffle(cast)
            refs = [{"name": x, "type": "human"} for x in cast]
            qsurf = question_surface(vq)
            qspec = {"type": "Q69", "surface": qsurf, "v": vq,
                     "a": cast[0], "p": cast[1]}
            for suffix, qa, qp, ans in (("a", b, n, "yes"), ("b", n, b, "no")):
                ev = [None, None, None]
                ev[qi - 1] = {"verb": slot_verbs[qi - 1],
                              "agent": qa, "patient": qp, "pro_drop": False}
                for k in range(3):
                    if k == qi - 1:
                        continue
                    ev[k] = {"verb": slot_verbs[k],
                             "agent": uq_agent, "patient": uq_patient,
                             "pro_drop": False}
                flags = [o1, o2, o3]
                items.append({
                    "id": "%s_%s_%s" % (split, key, suffix),
                    "pair_id": key, "split": split, "qtype": "Q69",
                    "hop_count": 1, "pro_drop_item": False,
                    "order_dependent": True, "answer": ans,
                    "K": 2, "T": 3, "flip_idx": [qi - 1],
                    "query_index": qi, "queried_verb": vq,
                    "verb_slots": {"V1": va, "V2": vb, "V3": vc},
                    "unqueried_bearer_is_agent": bool(uq_bearer_agent),
                    "referents": refs,
                    "events": ev,
                    "sentences": [render(flags[k], ev[k]["verb"],
                                         ev[k]["agent"], ev[k]["patient"])
                                  for k in (0, 1, 2)],
                    "order_flags": [o1, o2, o3],
                    "question": qspec,
                })

    # ---------------------------------------------------------------- invariants
    byT = {t: [i for i in items if i["T"] == t] for t in (1, 2, 3)}
    counts = {t: {s: sum(1 for i in byT[t] if i["split"] == s)
                  for s in ("train", "val", "test")} for t in (1, 2, 3)}

    # Budget check
    for t in (1, 2, 3):
        expected = N_CASTS * PAIRS_PER_CAST * 2
        assert len(byT[t]) == expected, \
            "T=%d budget is %d, expected %d" % (t, len(byT[t]), expected)
        assert counts[t]["test"] == N_TEST_ITEMS, \
            "T=%d n_test = %d, expected %d" % (t, counts[t]["test"], N_TEST_ITEMS)

    # Split integrity
    assert not (set(te_c) & (set(tr_c) | set(va_c))), "test casts leak"
    assert not (set(tr_c) & set(va_c)), "train/val casts overlap"
    assert len(te_c) + len(va_c) + len(tr_c) == len(casts)
    assert covers(tr_c, bearers, nonbearers), "train coverage failed"

    # Pair structure (generic: works at any T)
    byp = defaultdict(list)
    for i in items:
        byp[i["pair_id"]].append(i)
    for k, g in byp.items():
        assert len(g) == 2, "pair %s has %d items" % (k, len(g))
        a, b_ = sorted(g, key=lambda x: x["id"])
        qi = a["query_index"]
        assert a["answer"] != b_["answer"], k
        assert a["split"] == b_["split"], k
        assert a["query_index"] == b_["query_index"], k
        assert a["order_flags"] == b_["order_flags"], k
        assert a["question"] == b_["question"], k
        assert [r["name"] for r in a["referents"]] == \
               [r["name"] for r in b_["referents"]], k
        # Queried event: roles exchanged
        assert a["events"][qi - 1]["agent"] == b_["events"][qi - 1]["patient"], k
        assert a["events"][qi - 1]["patient"] == b_["events"][qi - 1]["agent"], k
        # Other events: byte-identical (the V8 guarantee + twin construction)
        for other in range(len(a["events"])):
            if other == qi - 1:
                continue
            assert a["events"][other] == b_["events"][other], \
                "event %d not byte-identical across twins of %s" % (other, k)
            assert a["sentences"][other] == b_["sentences"][other], k

    # Label assertions
    for i in items:
        qi = i["query_index"]
        assert {r["name"] for r in i["referents"]} <= set(INS.HUMANS), i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in bset) == 1, i["id"]
        assert (i["answer"] == "yes") == (i["events"][qi - 1]["agent"] in bset), \
            "label mismatch: %s" % i["id"]
        # V8: queried verb occurs in EXACTLY the queried event
        occ = [k for k, e in enumerate(i["events"]) if e["verb"] == i["queried_verb"]]
        assert occ == [qi - 1], \
            "queried verb not unique in queried event: %s" % i["id"]
        assert len(i["events"]) == i["T"] == len(i["sentences"]) == len(i["order_flags"])

    # T=3 slot violation check
    for i in [x for x in items if x["T"] == 3]:
        assert i["events"][0]["verb"] in set(V1), \
            "T=3 event[0] verb not in V1: %s" % i["id"]
        assert i["events"][1]["verb"] in set(V2), \
            "T=3 event[1] verb not in V2: %s" % i["id"]
        assert i["events"][2]["verb"] in set(V3), \
            "T=3 event[2] verb not in V3: %s" % i["id"]

    # Balance check (exact: pairs guarantee it)
    for t in (1, 2, 3):
        for s in ("train", "val", "test"):
            sub = [i for i in byT[t] if i["split"] == s]
            c = Counter(i["answer"] for i in sub)
            assert c["yes"] == c["no"], \
                "answer imbalance T=%d %s: yes=%d no=%d" % (t, s, c["yes"], c["no"])

    # ---- schedule assertions (structural, asserted not assumed)
    # T=2 verb-pair schedule: 80 casts × 6 pairs × 2 twins = 960 items; 4 verb combos
    # stride 5 coprime to 4 -> all 4 combos equally visited; 960/4 = 240 each
    vp2 = Counter("%s|%s" % (i["verb_slots"]["V1"], i["verb_slots"]["V2"])
                  for i in byT[2])
    assert set(vp2.values()) == {240}, \
        "T=2 verb-pair schedule not uniform (expected 240 each): %s" % dict(vp2)

    vp3 = Counter("%s|%s|%s" % (i["verb_slots"]["V1"], i["verb_slots"]["V2"],
                                  i["verb_slots"]["V3"])
                  for i in byT[3])
    assert set(vp3.values()) == {120}, \
        "T=3 verb-triple schedule not uniform (expected 120 each): %s" % dict(vp3)

    qi2 = Counter(str(i["query_index"]) for i in byT[2])
    assert set(qi2.values()) == {480}, \
        "T=2 query_index not balanced (expected 480 each): %s" % dict(qi2)

    qi3 = Counter(str(i["query_index"]) for i in byT[3])
    assert set(qi3.values()) == {320}, \
        "T=3 query_index not balanced (expected 320 each): %s" % dict(qi3)

    # ---------------------------------------------------------------- metadata
    meta = {
        "experiment": "exp85",
        "rung": "SB-DisCoCirc (Q69, T=3 extension)",
        "design_doc": SPEC,
        "seed": seed,
        "n_items": len(items),
        "T_values": [1, 2, 3],
        "split_counts": {str(t): counts[t] for t in (1, 2, 3)},
        "slot_assignment": slot_assignment,
        "slot_rule": (
            "2/2/2 split of the 6-verb exp51 inventory drawn at seed 85999; "
            "event i draws from slot i (1-indexed); disjointness ensures the "
            "verb-indexed query is well-posed (doc 55 §4.2) and costs any "
            "claim about discourse-position resolution (doc 55 §11)."),
        "verb_inventory": {v: {"human_patients": True} for v in verbs},
        "verb_inventory_source": VT_SRC,
        "category": {
            "key": "nonoccupational",
            "gloss": INS.CATEGORIES["nonoccupational"]["gloss"],
            "bearers": bearers,
            "nonbearers": nonbearers,
            "source": "instrument_sb_bounds.CATEGORIES -- doc 54 §5.2, imported",
        },
        "humans": list(INS.HUMANS),
        "n_casts": len(casts),
        "cast_pool_draws": pool_draws,
        "split_draws_until_satisfied": draws,
        "train_casts": [list(c) for c in tr_c],
        "val_casts": [list(c) for c in va_c],
        "test_casts": [list(c) for c in te_c],
        "split_unit": ("CAST -- a (bearer, non-bearer) pairing; SHARED across all T, "
                       "so the T=1/T=2/T=3 control compares like with like."),
        "heldout_is": "UNSEEN PAIRINGS (same rule as exp69, doc 55 §4.5).",
        "pairs_per_cast": PAIRS_PER_CAST,
        "schedule_rule": (
            "T=2: verb pair by stride 5 mod 4; surface-order pair by stride 3 mod 4. "
            "T=3: verb triple by stride 5 mod 8; surface-order triple by stride 3 mod 8; "
            "qpat [1,1,2,2,3,3] per-cast shuffle gives 2 items per query index. "
            "T=1: verb alternates V1[j%2] (3 each per cast). "
            "All strides coprime to their moduli."),
        "t3_schedule_counts": {
            "verb_triples": dict(vp3),
            "query_index": dict(qi3),
        },
        "marker_token": MARKER,
        "question_template": "الذي [v] هل هو [MARKER]",
        "invariants_asserted": [
            "V1, V2, V3 pairwise disjoint; union = 6-verb inventory",
            "T=3 event k draws from slot k+1 (0-indexed k)",
            "twin: roles exchanged in queried event; other events byte-identical",
            "label == yes iff agent of queried event is a bearer",
            "queried verb occurs in EXACTLY the queried event (V8)",
            "answer classes balanced in every split at every T",
            "n_test == 300 per T",
            "T=2 verb-pair schedule uniform (240 each over 4 combos)",
            "T=3 verb-triple schedule uniform (120 each over 8 combos)",
        ],
    }
    meta["dataset_sha256"] = comp.canonical_sha256(items)
    return items, meta


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--validate", action="store_true",
                    help="load %s and run slot-disjointness + SHA check" % OUT)
    a = ap.parse_args()

    if a.validate:
        blob = json.load(open(a.out, encoding="utf-8"))
        items, meta = blob["items"], blob["meta"]
        # SHA self-check
        sha = comp.canonical_sha256(items)
        assert sha == meta["dataset_sha256"], \
            "SHA mismatch: recomputed %s, stored %s" % (sha, meta["dataset_sha256"])
        # Slot disjointness
        V1 = set(meta["slot_assignment"]["V1"])
        V2 = set(meta["slot_assignment"]["V2"])
        V3 = set(meta["slot_assignment"]["V3"])
        assert not (V1 & V2) and not (V1 & V3) and not (V2 & V3), \
            "Slot disjointness violated"
        assert V1 | V2 | V3 == set(meta["verb_inventory"].keys()), \
            "V1 ∪ V2 ∪ V3 != full inventory"
        # T=3 slot assignments
        t3 = [i for i in items if i["T"] == 3]
        for i in t3:
            assert i["events"][0]["verb"] in V1, i["id"]
            assert i["events"][1]["verb"] in V2, i["id"]
            assert i["events"][2]["verb"] in V3, i["id"]
        print("[exp85 validate] SHA OK: %s" % sha)
        print("[exp85 validate] Slots: V1=%s V2=%s V3=%s" % (
            sorted(V1), sorted(V2), sorted(V3)))
        print("[exp85 validate] T=3 slot assignments: %d items checked -- OK" % len(t3))
        print("[exp85 validate] n_items: %d" % meta["n_items"])
        print("[exp85 validate] split_counts: %s" % meta["split_counts"])
        return 0

    print("[exp85_datagen] generating with seed %d..." % SEED, flush=True)
    items, meta = build(SEED)
    json.dump({"meta": meta, "items": items},
              open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[exp85_datagen] wrote %s" % a.out)
    print("  seed: %d" % meta["seed"])
    print("  slot assignment: V1=%s | V2=%s | V3=%s"
          % (meta["slot_assignment"]["V1"],
             meta["slot_assignment"]["V2"],
             meta["slot_assignment"]["V3"]))
    print("  n_items: %d" % meta["n_items"])
    print("  T_values: %s" % meta["T_values"])
    for t in (1, 2, 3):
        print("  T=%d splits: %s" % (t, meta["split_counts"][str(t)]))
    print("  dataset_sha256: %s" % meta["dataset_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
