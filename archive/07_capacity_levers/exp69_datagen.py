# -*- coding: utf-8 -*-
"""exp69_datagen.py -- doc 55 SS4: the Q69 generator, and doc 55 SS7.5's V-suite.

WHAT Q69 IS (doc 55 SS4.2, the REGISTERED form, and the only one)

    text:      event 1 = (v1, agent, patient)      v1 from slot inventory V_A
               event 2 = (v2, agent, patient)      v2 from slot inventory V_B
               each event rendered independently as SVO or VSO
    question:  "the one who [v_q] -- is he a [CATEGORY]?"      q in {1, 2}
    label:     1 iff agent(event q) is a category bearer
    twin:      exchange agent and patient WITHIN event q ONLY;
               event (3 - q) is held byte-identical; question surface identical

Four alternatives are DECLINED in doc 55 SS4.2 with their reasons (persistence-only
is solvable by a one-event reader; conjunction breaks 50/50 and destroys the exact
V1 = 0.500; XOR is content-blind, doc 49 SS10.4's trap; chain fails the twin
construction at n = 2). Q69 is the only candidate that simultaneously keeps SB's
exact twin construction, 50/50 balance BY CONSTRUCTION, and a read that requires
SELECTING the event by verb identity before reading that event's agent.

WHY V_A AND V_B ARE DISJOINT, and the scope it costs (doc 55 SS4.2, SS11). With
disjoint slot inventories the query resolves by VERB IDENTITY, not by discourse
position. exp69 therefore tests COMPOSITION over a two-event circuit; it does NOT
test discourse-position resolution, anaphora, or multi-hop reasoning, and may
never be reported as though it did. V8 asserts the resolvability rather than
arguing it.

THE T = 1 ARMS (doc 55 SS4.5). "The T = 1 arms use the same generator with event 2
suppressed and the query fixed to q = 1, giving an identical 960/300 budget." Read
literally as "delete event 2 from a T = 2 item" the construction collapses -- a
q = 2 pair whose event 2 is deleted has two BYTE-IDENTICAL twins carrying opposite
labels -- so the generator emits T = 1 items in their own mode: one event, q = 1,
six pairs per cast (ONE PER VERB, all six), which is exp68's construction on
exp69's surface and split. That is what makes R1-A a genuine replication target
for exp68's FA = 0.9187 and what makes doc 55 SS6.10(b)'s qualitative control
legitimate: T = 1 and T = 2 share the CAST SPLIT exactly, so a T = 2 failure with
T = 1 passing is located at composition and not at the encoding, the features or
the split.

THE QUESTION SURFACE is the same relative-clause template at both T (doc 55 SS18.1
closed the SS16.1 surface gate on it: grammatical, natural MSA, reading as a
question about the AGENT of the named event, with the verb resolving the query
unambiguously). Using one template at both T removes a nuisance difference from
SS6.10(b)'s control. The compiled circuit never sees the surface -- it consumes
per-referent angles and a question SPEC -- so the template matters to the V1/V2
surface baselines and to nothing exp69 measures.

TWO PARTS, ONE GENERATOR (doc 55 SS18.4 / SS19.1, Part B released 2026-08-25)

    Part A  category = NON-OCCUPATIONAL (4 bearers, 20 non-bearers), angles from
            the raw AraVec PCA plane. |B| x |N| = 80 casts exactly.
    Part B  category = faail vs non-faail (9 bearers, 15 non-bearers), angles from
            exp65's POSITIONAL featurizer. |B| x |N| = 135 possible casts, of
            which 80 are DRAWN at the registered seed so that the budget, the
            25-cast test split and therefore n_test = 300 match Part A exactly and
            doc 55 SS6's 0.06 bar (2 x the binomial SD at n = 300) transfers
            EXACTLY rather than approximately. doc 55 does not fix Part B's cast
            budget; mirroring Part A is a JUDGMENT CALL and is recorded as one in
            meta.part_b_budget_note.

MEMBERSHIP IS IMPORTED, NEVER RETYPED: Part A from
`instrument_sb_bounds.CATEGORIES["nonoccupational"]` (doc 54 SS5.2, user-confirmed
2026-08-23), Part B from `exp65_featurizer.FAAIL` (doc 51 SS7, user-confirmed).
doc 55 SS16.1: no new noun and no new verb is introduced, so the membership gate
stays closed; only the SURFACE reopened, and SS18.1 closed it.

BALANCE IS SCHEDULED, NOT SAMPLED (doc 55 SS7.5 V3). Verb pair, per-event surface
order, unqueried-event roles and query index are assigned from CYCLIC SCHEDULES
with strides chosen coprime to the cast count, so the marginals are balanced by
construction and no nuisance variable is perfectly confounded with the slot index.
Only the per-cast q-permutation and the per-pair wire shuffle are random. Every
balance claim is then RE-MEASURED in the V-suite, never assumed.

COVERAGE IS ASSERTED (doc 55 SS4.5): every bearer and every non-bearer in >= 1
TRAIN cast; every verb in V_A and in V_B in >= 1 train item; both surface orders
for EACH event slot in train; both query values in train and test in exactly equal
proportion. The generator REDRAWS from the registered seed's own stream until the
rule holds and records how many draws that took.

ORDERING (doc 55 SS19.2 / SS19.7): this script REFUSES TO RUN unless
`exp69_predictions.json` already exists. The predictions are frozen before the
dataset exists, and that is mechanised here rather than remembered.

    python exp69_datagen.py                     # both parts, seed 69000
    python exp69_datagen.py --seed 70000        # doc 55 SS4.5's CONFIRM draw
    python exp69_datagen.py --validate          # doc 55 SS7.5's V-suite kill-gate
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

import exp41_story_generator as gen
import exp42_compiler as comp
import exp65_featurizer as WZ
import instrument_sb_bounds as INS

SPEC = "qnlp_private_docs/55_exp69_reunion.md (da4b8fe, REGISTERED through SS19.7)"
VT_SRC = "stories_exp51_VT.json"
PREDICTIONS = "exp69_predictions.json"
OUT_A = "stories_exp69_QA.json"          # Part A: Q69 x non-occupational
OUT_B = "stories_exp69_QB.json"          # Part B: Q69 x wazn (faail)
VALIDATION_OUT = "results_exp69_WN_validation.json"

SEED = 69000                             # registered generator seed
CONFIRM_SEED = SEED + 1000               # doc 46 SS10.4 / doc 55 SS4.5
N_CASTS = 80                             # both parts, doc 55 SS4.5
N_TEST_CASTS = 25                        # -> 150 pairs -> 300 items, per T
VAL_FRACTION = 0.20                      # by CAST, never within a pair
PAIRS_PER_CAST = 6                       # doc 55 SS4.5
MARKER = "المميز"                          # placeholder; names no referent
MAX_DRAWS = 500
N_TEST_ITEMS = 300                       # doc 55 SS4.5, frozen

V6_TRAIN_BAR = 0.80                      # exp68_validate's functional-probe bar


# ============================================================ the two parts
def part_spec(part):
    """Bearers / non-bearers / angle map for one part. IMPORTED, never retyped."""
    humans = list(INS.HUMANS)
    assert sorted(humans) == sorted(gen.HUMANS), \
        "instrument inventory != generator inventory -- population drift"
    if part == "A":
        bearers = sorted(INS.CATEGORIES["nonoccupational"]["members"])
        nonbearers = sorted(n for n in humans if n not in set(bearers))
        assert (len(bearers), len(nonbearers)) == (4, 20), \
            "doc 54 SS5.2 membership drift"
        return {"part": "A", "key": "nonoccupational",
                "gloss": INS.CATEGORIES["nonoccupational"]["gloss"],
                "bearers": bearers, "nonbearers": nonbearers,
                "glosses": {m: INS.GLOSS[m] for m in bearers},
                "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                           "user-confirmed 2026-08-23; imported, never retyped"),
                "feature_family": "raw AraVec PCA plane (FA angle / F2 register)"}
    bearers = sorted(n for n in humans if n in WZ.FAAIL)
    nonbearers = sorted(n for n in humans if n not in WZ.FAAIL)
    assert (len(bearers), len(nonbearers)) == (9, 15), \
        "doc 51 SS7 faail membership drift"
    return {"part": "B", "key": "wazn_faail",
            "gloss": "faail (CaaCiC) wazn vs non-faail",
            "bearers": bearers, "nonbearers": nonbearers,
            "glosses": {m: INS.GLOSS[m] for m in bearers},
            "source": ("exp65_featurizer.FAAIL -- doc 51 SS7, user-confirmed; "
                       "imported, never retyped"),
            "feature_family": ("exp65 POSITIONAL featurizer angle, fit "
                               "INVENTORY-WIDE (doc 55 SS18.4: not leakage here "
                               "because exp69 holds out PAIRINGS, not nouns)")}


# ================================================================ surfaces
def render(order, verb, agent, patient):
    """The two orders the archived pregroup reader implements
    (`archive/01_parser_pipeline/arabic_dep_reader.py:130-138`, SVO
    `Word(v, N.r @ S @ N.l)` and VSO `Word(v, S @ N.l @ N.l)`)."""
    if order == "SVO":
        return "%s %s %s" % (agent, verb, patient)
    return "%s %s %s" % (verb, agent, patient)


def question_surface(vq):
    """doc 55 SS4.2 / SS18.1: 'the one who [v] -- is he a [CATEGORY]?'. The marker
    names no referent, so both twins share the surface EXACTLY."""
    return "الذي %s هل هو %s" % (vq, MARKER)


# ==================================================================== split
def draw_cast_pool(rng, bearers, nonbearers):
    """All (b, n) pairings, then N_CASTS of them. Part A has exactly 80 and the
    draw is the identity; Part B has 135 and the 80 are DRAWN at the registered
    seed with a coverage acceptance rule of its own."""
    allc = [(b, n) for b in bearers for n in nonbearers]
    if len(allc) == N_CASTS:
        return sorted(allc), 0
    draws = 0
    while True:
        draws += 1
        if draws > MAX_DRAWS:
            raise RuntimeError("cast-pool coverage never satisfied in %d draws "
                               "-- STOP, do not weaken the rule" % MAX_DRAWS)
        pool = list(allc)
        rng.shuffle(pool)
        cand = sorted(pool[:N_CASTS])
        seen = {x for c in cand for x in c}
        if set(bearers) <= seen and set(nonbearers) <= seen:
            return cand, draws


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
def build(part, seed=SEED):
    spec = part_spec(part)
    rng = random.Random(seed + (0 if part == "A" else 500))
    bearers, nonbearers = spec["bearers"], spec["nonbearers"]

    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())
    assert len(verbs) == 6, "doc 55 SS4.1: the 6-verb exp51 inventory, unchanged"
    vshuf = list(verbs)
    rng.shuffle(vshuf)                    # the registered 3/3 slot partition
    V_A, V_B = sorted(vshuf[:3]), sorted(vshuf[3:])
    assert not (set(V_A) & set(V_B)), "V_A and V_B must be DISJOINT (doc 55 SS4.2)"
    VPAIRS = [(a, b) for a in V_A for b in V_B]          # 9 combinations
    ORDERS = [("SVO", "SVO"), ("SVO", "VSO"), ("VSO", "SVO"), ("VSO", "VSO")]

    casts, pool_draws = draw_cast_pool(rng, bearers, nonbearers)
    assert len(casts) == N_CASTS

    draws = 0
    while True:
        draws += 1
        if draws > MAX_DRAWS:
            raise RuntimeError("train coverage never satisfied in %d draws at "
                               "seed %d -- STOP, do not weaken the rule"
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

    items, pid1, pid2 = [], 0, 0
    for ci, cast_key in enumerate(casts):
        b, n = cast_key
        split = split_of[cast_key]

        # ---------------- T = 1: one pair per verb (exp68's construction)
        for j, v in enumerate(verbs):
            pid1 += 1
            key = "p1_%05d" % pid1
            cast = [b, n]
            rng.shuffle(cast)                 # bearer not systematically wire 0
            refs = [{"name": x, "type": "human"} for x in cast]
            order = ["SVO", "VSO"][(ci + j) % 2]   # exact 40/40 per verb over casts
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
                    "verb_slots": {"V_A": v, "V_B": None},
                    "referents": refs,
                    "events": [{"verb": v, "agent": agent, "patient": patient,
                                "pro_drop": False}],
                    "sentences": [render(order, v, agent, patient)],
                    "order_flags": [order],
                    "question": q,
                })

        # ---------------- T = 2: six Q69 pairs, three at q = 1 and three at q = 2
        qpat = [1, 1, 1, 2, 2, 2]
        rng.shuffle(qpat)
        for j in range(PAIRS_PER_CAST):
            pid2 += 1
            key = "p2_%05d" % pid2
            va, vb = VPAIRS[(5 * ci + j) % 9]          # stride 5, coprime to 9
            o1, o2 = ORDERS[(3 * ci + j) % 4]          # stride 3, coprime to 4
            uq_bearer_agent = ((ci + j) % 2 == 0)      # unqueried-event roles
            qi = qpat[j]
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
                flags = [o1, o2]
                items.append({
                    "id": "%s_%s_%s" % (split, key, suffix),
                    "pair_id": key, "split": split, "qtype": "Q69",
                    "hop_count": 1, "pro_drop_item": False,
                    "order_dependent": True, "answer": ans,
                    "K": 2, "T": 2, "flip_idx": [qi - 1],
                    "query_index": qi, "queried_verb": vq,
                    "verb_slots": {"V_A": va, "V_B": vb},
                    "unqueried_bearer_is_agent": bool(uq_bearer_agent),
                    "referents": refs,
                    "events": ev,
                    "sentences": [render(flags[k], ev[k]["verb"],
                                         ev[k]["agent"], ev[k]["patient"])
                                  for k in (0, 1)],
                    "order_flags": flags,
                    "question": qspec,
                })

    bset = set(bearers)
    byT = {t: [i for i in items if i["T"] == t] for t in (1, 2)}
    counts = {t: {s: sum(1 for i in byT[t] if i["split"] == s)
                  for s in ("train", "val", "test")} for t in (1, 2)}

    # ------------------------------------- invariants, asserted BEFORE writing
    assert not (set(te_c) & (set(tr_c) | set(va_c))), "test casts leak into train/val"
    assert not (set(tr_c) & set(va_c)), "train and val casts overlap"
    assert len(te_c) + len(va_c) + len(tr_c) == len(casts)
    assert covers(tr_c, bearers, nonbearers), "train coverage assertion failed"
    for t in (1, 2):
        assert len(byT[t]) == N_CASTS * PAIRS_PER_CAST * 2 == 960, \
            "T=%d budget is %d, not the registered 960" % (t, len(byT[t]))
        assert counts[t]["test"] == N_TEST_ITEMS, \
            "T=%d n_test = %d, not the registered 300" % (t, counts[t]["test"])

    byp = defaultdict(list)
    for i in items:
        byp[i["pair_id"]].append(i)
    for k, g in byp.items():
        assert len(g) == 2, k
        a, b_ = sorted(g, key=lambda x: x["id"])
        qi = a["query_index"]
        assert a["answer"] != b_["answer"], k
        assert a["split"] == b_["split"], k
        assert a["query_index"] == b_["query_index"], k
        assert a["order_flags"] == b_["order_flags"], k
        assert a["question"] == b_["question"], k
        assert [r["name"] for r in a["referents"]] == \
               [r["name"] for r in b_["referents"]], k
        # roles exchanged in the QUERIED event only
        assert a["events"][qi - 1]["agent"] == b_["events"][qi - 1]["patient"], k
        assert a["events"][qi - 1]["patient"] == b_["events"][qi - 1]["agent"], k
        # the OTHER event is byte-identical (doc 55 SS4.2)
        for other in range(len(a["events"])):
            if other == qi - 1:
                continue
            assert a["events"][other] == b_["events"][other], \
                "event %d is not byte-identical across twins of %s" % (other, k)
            assert a["sentences"][other] == b_["sentences"][other], k

    for i in items:
        qi = i["query_index"]
        assert {r["name"] for r in i["referents"]} <= set(INS.HUMANS), i["id"]
        assert sum(1 for r in i["referents"] if r["name"] in bset) == 1, i["id"]
        assert (i["answer"] == "yes") == (i["events"][qi - 1]["agent"] in bset), i["id"]
        # V8, asserted in the generator too (doc 55 SS7.5)
        occ = [k for k, e in enumerate(i["events"]) if e["verb"] == i["queried_verb"]]
        assert occ == [qi - 1], \
            "queried verb does not occur in exactly the queried event: %s" % i["id"]
        assert len(i["events"]) == i["T"] == len(i["sentences"]) == len(i["order_flags"])

    bal = {t: {s: {a: sum(1 for i in byT[t]
                          if i["split"] == s and i["answer"] == a)
                   for a in ("yes", "no")}
               for s in ("train", "val", "test")} for t in (1, 2)}
    for t in (1, 2):
        for s, c in bal[t].items():
            assert c["yes"] == c["no"], "answer imbalance T=%d %s: %s" % (t, s, c)

    # ------------------------------------------------- coverage of the design
    tr_items = [i for i in items if i["split"] == "train"]
    tr2 = [i for i in tr_items if i["T"] == 2]
    cov = {
        "verbs_VA_in_train": sorted({i["events"][0]["verb"] for i in tr2}),
        "verbs_VB_in_train": sorted({i["events"][1]["verb"] for i in tr2}),
        "orders_event1_in_train": sorted({i["order_flags"][0] for i in tr2}),
        "orders_event2_in_train": sorted({i["order_flags"][1] for i in tr2}),
        "query_values_in_train": sorted({i["query_index"] for i in tr2}),
        "query_values_in_test": sorted({i["query_index"] for i in items
                                        if i["split"] == "test" and i["T"] == 2}),
        "verbs_T1_in_train": sorted({i["events"][0]["verb"]
                                     for i in tr_items if i["T"] == 1}),
    }
    assert set(cov["verbs_VA_in_train"]) == set(V_A), "V_A not covered in train"
    assert set(cov["verbs_VB_in_train"]) == set(V_B), "V_B not covered in train"
    assert cov["orders_event1_in_train"] == ["SVO", "VSO"], "event-1 order not covered"
    assert cov["orders_event2_in_train"] == ["SVO", "VSO"], "event-2 order not covered"
    assert cov["query_values_in_train"] == [1, 2], "query values not covered in train"
    assert cov["query_values_in_test"] == [1, 2], "query values not covered in test"
    assert set(cov["verbs_T1_in_train"]) == set(verbs), "T=1 verbs not covered"
    for s in ("train", "test"):
        qc = Counter(i["query_index"] for i in items if i["split"] == s and i["T"] == 2)
        assert qc[1] == qc[2], \
            "query values not in equal proportion in %s: %s" % (s, dict(qc))

    schedules = {
        "verb_pair": dict(Counter("%s|%s" % (i["verb_slots"]["V_A"],
                                             i["verb_slots"]["V_B"])
                                  for i in byT[2])),
        "order_pair": dict(Counter("%s|%s" % tuple(i["order_flags"])
                                   for i in byT[2])),
        "query_index": dict(Counter(str(i["query_index"]) for i in byT[2])),
        "unqueried_bearer_is_agent": dict(Counter(
            str(i["unqueried_bearer_is_agent"]) for i in byT[2])),
        "T1_verb": dict(Counter(i["events"][0]["verb"] for i in byT[1])),
        "T1_order": dict(Counter(i["order_flags"][0] for i in byT[1])),
    }
    # the schedules are cyclic, so these are structural facts, asserted
    assert set(schedules["order_pair"].values()) == {240}, schedules["order_pair"]
    assert set(schedules["query_index"].values()) == {480}, schedules["query_index"]
    assert set(schedules["unqueried_bearer_is_agent"].values()) == {480}
    assert set(schedules["T1_verb"].values()) == {160}, schedules["T1_verb"]
    assert set(schedules["T1_order"].values()) == {480}, schedules["T1_order"]
    vp = list(schedules["verb_pair"].values())
    assert len(vp) == 9 and max(vp) - min(vp) <= 4, \
        ("verb-pair schedule is not near-uniform: %s. 960 T=2 items over 9 "
         "combinations cannot be EXACTLY uniform (960/9 = 106.67); the bound is "
         "recorded rather than the imbalance being hidden." % schedules["verb_pair"])

    meta = {
        "experiment": "exp69", "part": spec["part"], "rung": "SB-DisCoCirc (Q69)",
        "design_doc": SPEC,
        "seed": seed, "is_confirm_draw": bool(seed == CONFIRM_SEED),
        "n_items": len(items), "split_counts": {str(t): counts[t] for t in (1, 2)},
        "T_values": [1, 2],
        "verb_inventory": {v: {"human_patients": True} for v in verbs},
        "verb_inventory_source": VT_SRC,
        "verb_slots": {"V_A": V_A, "V_B": V_B,
                       "rule": ("the 6-verb exp51 inventory split 3/3 into DISJOINT "
                                "slots at the registered generator seed; event 1 "
                                "draws from V_A, event 2 from V_B (doc 55 SS4.2). "
                                "Disjointness is what makes the verb-indexed query "
                                "well-posed and is what costs exp69 any claim about "
                                "discourse-position resolution (doc 55 SS11).")},
        "humans": list(INS.HUMANS), "objects": [],
        "category": {"key": spec["key"], "gloss": spec["gloss"],
                     "source": spec["source"], "glosses": spec["glosses"],
                     "feature_family": spec["feature_family"]},
        "bearers": bearers, "nonbearers": nonbearers,
        "n_casts": len(casts),
        "cast_pool_draws": pool_draws,
        "part_b_budget_note": (
            "Part B has 9 x 15 = 135 possible casts; 80 are DRAWN at the registered "
            "seed so the budget, the 25-cast test split and n_test = 300 match Part "
            "A exactly and doc 55 SS6's 0.06 bar transfers exactly. doc 55 SS18.4 "
            "does not fix Part B's cast budget, so this is a JUDGMENT CALL recorded "
            "as one." if spec["part"] == "B" else
            "Part A has exactly |B| x |N| = 4 x 20 = 80 casts; no draw is needed."),
        "train_casts": [list(c) for c in tr_c],
        "val_casts": [list(c) for c in va_c],
        "test_casts": [list(c) for c in te_c],
        "split_unit": ("CAST -- a (bearer, non-bearer) pairing; never within a pair, "
                       "and SHARED between the T = 1 and T = 2 item sets so doc 55 "
                       "SS6.10(b)'s qualitative control compares like with like"),
        "heldout_is": (
            "UNSEEN PAIRINGS, not unseen nouns (doc 55 SS4.5). Every noun's feature "
            "is in-inventory by construction, which is what makes the inventory-wide "
            "PCA fit (Part A) and the inventory-wide positional-featurizer fit "
            "(Part B, doc 55 SS18.4) legitimate, and what makes doc 55 SS11's scope "
            "limit binding: exp69 does NOT test novel-filler generalisation."),
        "coverage_assertion": {
            "rule": ("every bearer AND every non-bearer in >= 1 TRAIN cast; every "
                     "verb of V_A and V_B in >= 1 train item; both surface orders "
                     "for EACH event slot in train; both query values in train and "
                     "test in exactly equal proportion (doc 55 SS4.5)"),
            "draws_until_satisfied": draws,
            "nouns_in_train_casts": len({n for c in tr_c for n in c}),
            "coverage": cov, "satisfied": True},
        "balance_schedules": schedules,
        "schedule_rule": (
            "verb pair by stride 5 mod 9, surface-order pair by stride 3 mod 4, "
            "unqueried-event roles by (cast + slot) mod 2, query index by a "
            "per-cast permutation of [1,1,1,2,2,2]. Strides are coprime to their "
            "moduli so no nuisance variable is confounded with the slot index, and "
            "every marginal is RE-MEASURED in the V-suite rather than assumed."),
        "marker_token": MARKER,
        "question_template": ("الذي [v] هل هو [MARKER] -- 'the one who [v] -- is he "
                              "a [CATEGORY]?'. doc 55 SS16.1's three surface "
                              "questions were answered in the affirmative by the "
                              "user (doc 55 SS18.1, verbatim 'yes to all three'), "
                              "which is what released datagen. The SAME template is "
                              "used at T = 1 and T = 2 so the SS6.10(b) control "
                              "differs only in the number of events."),
        "answer_balance": {str(t): bal[t] for t in (1, 2)},
        "construction": (
            "Q69 (doc 55 SS4.2): a text of T events over ONE two-referent cast with "
            "exactly one category bearer, followed by a question NAMING ONE OF THE "
            "VERBS. The label is 1 iff the agent of the NAMED event is a bearer. The "
            "twin exchanges agent and patient WITHIN the queried event only; the "
            "other event is byte-identical and the question surface is identical, so "
            "any function of the cast alone scores EXACTLY 0.500 and the classes are "
            "50/50 by construction. Each event is rendered independently as SVO or "
            "VSO -- the alternation the archived pregroup reader implements "
            "(archive/01_parser_pipeline/arabic_dep_reader.py:130-138, 167-180)."),
        "invariants_asserted": [
            "test casts disjoint from train and val casts (unseen pairings)",
            "the T = 1 and T = 2 item sets share ONE cast split",
            "every bearer and every non-bearer appears in >= 1 train cast",
            "every pair: identical cast, identical wire order, identical question "
            "spec, identical surface order flags, opposite answers",
            "roles exchanged in the QUERIED event only; the other event is "
            "byte-identical across twins, sentence included",
            "exactly one bearer in every cast",
            "answer == yes iff the agent of the QUERIED event is a real bearer",
            "the queried verb occurs in EXACTLY ONE event (V8, doc 55 SS7.5)",
            "answer classes balanced in every split at every T",
            "n_test == 300 at each T, matching exp60/exp61/exp68 so doc 55 SS6's "
            "0.06 bar transfers exactly",
            "V_A and V_B are disjoint",
        ],
    }
    meta["dataset_sha256"] = comp.canonical_sha256(items)
    return items, meta


# ======================================================== doc 55 SS7.5 V-suite
def _split(items, s):
    return [i for i in items if i["split"] == s]


def _acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def angle_map(meta):
    """The arm's own per-referent angle, for V1. Part A uses the frozen exp42 FA
    map; Part B uses exp65's positional featurizer fit inventory-wide."""
    if meta["part"] == "A":
        base, _m, _s = comp.load_embeddings()
        return {n: float(base[n]) for n in INS.HUMANS}, "exp42_noun_embeddings (FA)"
    H = sorted(gen.HUMANS)
    return ({n: float(v) for n, v in WZ.angles_for(H, INS.HUMANS).items()},
            "exp65_featurizer positional angle, inventory-wide fit")


def v1_bag(items, meta):
    """doc 55 SS7.5: EXACTLY 0.500 with twin features BIT-IDENTICAL. Re-measured
    at multi-event scale, never inherited -- doc 46 SS8's lesson."""
    from sklearn.linear_model import LogisticRegression
    base, src = angle_map(meta)

    def feat(it):
        v = sorted(base[r["name"]] for r in it["referents"])
        return [np.cos(v[0]), np.sin(v[0]), np.cos(v[1]), np.sin(v[1]),
                np.cos(v[0]) + np.cos(v[1]), np.sin(v[0]) + np.sin(v[1])]

    out = {"angle_source": src}
    for t in (1, 2):
        sub = [i for i in items if i["T"] == t]
        tr, te = _split(sub, "train"), _split(sub, "test")
        Xtr = np.array([feat(i) for i in tr])
        Xte = np.array([feat(i) for i in te])
        m = LogisticRegression(max_iter=2000).fit(Xtr, [i["answer"] for i in tr])
        a = _acc([i["answer"] for i in te], m.predict(Xte))
        byp = defaultdict(list)
        for k, it in enumerate(te):
            byp[it["pair_id"]].append(k)
        bit = all(np.array_equal(Xte[x], Xte[y]) for x, y in byp.values())
        out["T%d" % t] = {"test_accuracy": a,
                          "twin_features_bit_identical": bool(bit),
                          "n_twin_pairs_checked": len(byp),
                          "passed": bool(a == 0.5 and bit)}
    out["passed"] = bool(all(out["T%d" % t]["passed"] for t in (1, 2)))
    return out


def _tokens(it):
    return " ".join(it["sentences"] + [it["question"]["surface"]])


def v2_ngram(items):
    """RE-DERIVED and reported UN-GATED, exactly as exp68 did. The reason is
    unchanged and is now doc 55 SS7.5's: under a held-out-PAIRING split the same 24
    nouns appear in train and test and the surface names both referents, so token
    identity transfers BY DESIGN and exp60's V2-H bar of 0.55 would fail the
    dataset FOR SATISFYING ITS OWN REGISTRATION. The price is paid in the CLAIM
    (doc 55 SS11): exp69 is not novel-filler generalisation."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    out = {"gated": False, "exp60_V2H_bar_not_inherited": 0.55,
           "why_no_bar": (
               "doc 55 SS7.5, inherited from exp68's rederivation_note: a pairing "
               "split shares the noun inventory across splits and the surface names "
               "both referents, so an n-gram model transfers to unseen pairings by "
               "design. The compiled circuit consumes per-referent angles only and "
               "never sees a token, so the shortcut is available to surface "
               "baselines and to nothing exp69 measures.")}
    for t in (1, 2):
        sub = [i for i in items if i["T"] == t]
        tr, va, te = (_split(sub, s) for s in ("train", "val", "test"))
        vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+")
        Xtr = vec.fit_transform([_tokens(i) for i in tr])
        m = LogisticRegression(max_iter=3000, C=10.0).fit(
            Xtr, [i["answer"] for i in tr])
        out["T%d" % t] = {
            "seen_pairings_val_accuracy": _acc(
                [i["answer"] for i in va],
                m.predict(vec.transform([_tokens(i) for i in va]))),
            "unseen_pairings_test_accuracy": _acc(
                [i["answer"] for i in te],
                m.predict(vec.transform([_tokens(i) for i in te]))),
            "n_features": int(Xtr.shape[1])}
    return out


def v3_balance(items, meta):
    """doc 55 SS7.5: verb, VERB-PAIR, SURFACE-ORDER, QUERY-INDEX, bearer-wire and
    wire-0 noun all balanced. The agent-noun imbalance is DESCRIPTIVE -- it is the
    task."""
    from scipy.stats import chi2_contingency
    bearers = set(meta["bearers"])
    out, worst = {}, 1.0

    def cells(sub, keyfn):
        t = defaultdict(Counter)
        for i in sub:
            t[keyfn(i)][i["answer"]] += 1
        rows = [[c["yes"], c["no"]] for c in t.values() if sum(c.values()) > 0]
        if len(rows) < 2:
            return 1.0
        return float(chi2_contingency(np.array(rows))[1])

    for t in (1, 2):
        sub = [i for i in items if i["T"] == t]
        checks = {
            "queried_verb": lambda i: i["queried_verb"],
            "event1_verb": lambda i: i["events"][0]["verb"],
            "bearer_wire": lambda i: 0 if i["referents"][0]["name"] in bearers else 1,
            "order_event1": lambda i: i["order_flags"][0],
            "wire0_noun": lambda i: i["referents"][0]["name"],
        }
        if t == 2:
            checks.update({
                "event2_verb": lambda i: i["events"][1]["verb"],
                "verb_pair": lambda i: "%s|%s" % (i["verb_slots"]["V_A"],
                                                  i["verb_slots"]["V_B"]),
                "order_event2": lambda i: i["order_flags"][1],
                "order_pair": lambda i: "%s|%s" % tuple(i["order_flags"]),
                "query_index": lambda i: i["query_index"],
                "unqueried_roles": lambda i: i["unqueried_bearer_is_agent"],
            })
        cell = {}
        for k, f in checks.items():
            p = cells(sub, f)
            cell[k + "_p"] = p
            worst = min(worst, p)
        cell["agent_noun_p_DESCRIPTIVE"] = cells(
            sub, lambda i: i["events"][i["query_index"] - 1]["agent"])
        c = Counter(i["answer"] for i in sub)
        cell.update({"n_yes": c["yes"], "n_no": c["no"],
                     "classes_balanced": c["yes"] == c["no"]})
        out["T%d" % t] = cell
    out["min_p"] = worst
    out["marginal_counts"] = meta["balance_schedules"]
    out["passed"] = bool(worst > 0.01
                         and all(out["T%d" % t]["classes_balanced"] for t in (1, 2)))
    return out


def v4_dedup(items):
    """doc 55 SS7.5: all 960 compiled circuits distinct at each T, zero
    contradictory label pairs. Dedup on what the CIRCUIT sees -- wires, the ordered
    event list and the question spec -- not on the surface."""
    out = {}
    for t in (1, 2):
        sub = [i for i in items if i["T"] == t]
        seen = defaultdict(set)
        for i in sub:
            w = {r["name"]: k for k, r in enumerate(i["referents"])}
            key = (tuple(r["name"] for r in i["referents"]),
                   tuple((e["verb"], w[e["agent"]], w[e["patient"]])
                         for e in i["events"]),
                   i["question"]["v"], w[i["question"]["a"]], w[i["question"]["p"]])
            seen[key].add(i["answer"])
        dup = sum(1 for v in seen.values() if len(v) > 1)
        out["T%d" % t] = {"n_distinct_circuits": len(seen), "n_items": len(sub),
                          "n_contradictory": dup,
                          "all_circuits_distinct": bool(len(seen) == len(sub)),
                          "passed": bool(dup == 0 and len(seen) == len(sub))}
    out["passed"] = bool(all(out["T%d" % t]["passed"] for t in (1, 2)))
    return out


def v5_split_integrity(items, meta):
    cast = lambda i: tuple(sorted(r["name"] for r in i["referents"]))  # noqa: E731
    by = {s: {cast(i) for i in _split(items, s)} for s in ("train", "val", "test")}
    disjoint = (not (by["test"] & (by["train"] | by["val"]))
                and not (by["train"] & by["val"]))
    in_train = {n for c in by["train"] for n in c}
    bearers, nons = set(meta["bearers"]), set(meta["nonbearers"])
    missing = sorted((bearers | nons) - in_train)
    n_test = {t: len([i for i in items if i["T"] == t and i["split"] == "test"])
              for t in (1, 2)}
    # the T = 1 and T = 2 sets must sit on the SAME split
    same = all({cast(i) for i in items if i["T"] == 1 and i["split"] == s}
               == {cast(i) for i in items if i["T"] == 2 and i["split"] == s}
               for s in ("train", "val", "test"))
    return {"n_train_casts": len(by["train"]), "n_val_casts": len(by["val"]),
            "n_test_casts": len(by["test"]),
            "casts_disjoint_across_splits": bool(disjoint),
            "bearers_covered_in_train": bool(bearers <= in_train),
            "nonbearers_covered_in_train": bool(nons <= in_train),
            "referents_missing_from_train_casts": missing,
            "n_test_items": {str(k): v for k, v in n_test.items()},
            "n_test_items_is_registered_300": bool(
                all(v == N_TEST_ITEMS for v in n_test.values())),
            "T1_and_T2_share_one_split": bool(same),
            "coverage_of_the_design": meta["coverage_assertion"]["coverage"],
            "why_coverage_matters": (
                "without it, 'unseen pairing' degenerates into 'unseen noun' for any "
                "referent absent from every train cast, and the split stops measuring "
                "what doc 55 SS4.5 says it measures"),
            "passed": bool(disjoint and not missing and same
                           and all(v == N_TEST_ITEMS for v in n_test.values()))}


def v6_token_probe(items, meta):
    """Noun-identity x role probe on the QUERIED event. Gated on the
    in-distribution half only -- a probe that cannot fit the seen pairings
    certifies nothing about the unseen ones. The held-out half is expected to
    succeed under a pairing split and is recorded with NO BAR."""
    from sklearn.linear_model import LogisticRegression
    nouns = sorted(meta["humans"])
    idx = {n: k for k, n in enumerate(nouns)}

    def feat(it):
        e = it["events"][it["query_index"] - 1]
        v = np.zeros(2 * len(nouns))
        v[idx[e["agent"]]] = 1
        v[len(nouns) + idx[e["patient"]]] = 1
        return v

    out = {"heldout_gated": False, "train_bar": V6_TRAIN_BAR,
           "exp60_heldout_bar_not_inherited": 0.55,
           "why_heldout_not_gated": (
               "the noun inventory is shared across splits by construction, so a "
               "noun-identity probe transfers to unseen PAIRINGS. That is a property "
               "of the registered split, not a defect, and it is exactly why doc 55 "
               "SS11 forbids the novel-filler reading.")}
    ok = True
    for t in (1, 2):
        sub = [i for i in items if i["T"] == t]
        tr, va, te = (_split(sub, s) for s in ("train", "val", "test"))
        m = LogisticRegression(max_iter=4000, C=10.0).fit(
            np.array([feat(i) for i in tr]), [i["answer"] for i in tr])
        av = _acc([i["answer"] for i in va], m.predict(np.array([feat(i) for i in va])))
        ah = _acc([i["answer"] for i in te], m.predict(np.array([feat(i) for i in te])))
        out["T%d" % t] = {"seen_pairings_val_accuracy": av,
                          "unseen_pairings_test_accuracy": ah,
                          "must_succeed_on_seen_pairings": bool(av >= V6_TRAIN_BAR)}
        ok = ok and av >= V6_TRAIN_BAR
    out["passed"] = bool(ok)
    return out


def v7_label_definition(items, meta):
    bset = set(meta["bearers"])
    bad = [i["id"] for i in items
           if (i["answer"] == "yes")
           != (i["events"][i["query_index"] - 1]["agent"] in bset)]
    one = [i["id"] for i in items
           if sum(1 for r in i["referents"] if r["name"] in bset) != 1]
    bal = {"T%d" % t: {s: dict(Counter(i["answer"] for i in items
                                       if i["T"] == t and i["split"] == s))
                       for s in ("train", "val", "test")} for t in (1, 2)}
    balanced = all(c["yes"] == c["no"] for d in bal.values() for c in d.values())
    return {"rule": "answer == yes iff the agent of the QUERIED event is a bearer",
            "items_violating_label_rule": bad[:8], "n_violations": len(bad),
            "items_without_exactly_one_bearer": one[:8], "n_cast_violations": len(one),
            "per_split_balance": bal, "balanced_in_every_split": bool(balanced),
            "passed": bool(not bad and not one and balanced)}


def v8_query_resolvability(items, meta):
    """NEW (doc 55 SS7.5): for every item the queried verb occurs in EXACTLY ONE of
    the events. This is what makes Q69 well-posed, and it is asserted rather than
    argued."""
    bad, wrong_event, cross = [], [], []
    VA, VB = set(meta["verb_slots"]["V_A"]), set(meta["verb_slots"]["V_B"])
    for i in items:
        occ = [k for k, e in enumerate(i["events"]) if e["verb"] == i["queried_verb"]]
        if len(occ) != 1:
            bad.append(i["id"])
        elif occ[0] != i["query_index"] - 1:
            wrong_event.append(i["id"])
        if i["T"] == 2:
            if i["events"][0]["verb"] not in VA or i["events"][1]["verb"] not in VB:
                cross.append(i["id"])
        if i["question"]["v"] != i["queried_verb"]:
            bad.append(i["id"])
    return {"n_items": len(items),
            "n_queried_verb_not_unique": len(bad),
            "n_queried_verb_in_wrong_event": len(wrong_event),
            "n_slot_violations": len(cross),
            "examples": (bad + wrong_event + cross)[:8],
            "slots_disjoint": bool(not (VA & VB)),
            "why": ("with DISJOINT slot inventories the query resolves by VERB "
                    "IDENTITY, not by discourse position (doc 55 SS4.2). That is what "
                    "makes Q69 well-posed and what costs exp69 any claim about "
                    "discourse-position resolution, anaphora or multi-hop reasoning "
                    "(doc 55 SS11)."),
            "passed": bool(not bad and not wrong_event and not cross and not (VA & VB))}


def validate(path):
    blob = json.load(open(path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("dataset self-hash mismatch -- refusing to validate")
    r = {"dataset": path, "part": meta["part"],
         "dataset_sha256": meta["dataset_sha256"], "dataset_seed": meta["seed"],
         "category": meta["category"], "bearers": meta["bearers"],
         "verb_slots": meta["verb_slots"], "split_counts": meta["split_counts"],
         "split_unit": meta["split_unit"],
         "V1_bag_of_embeddings": v1_bag(items, meta),
         "V2_ngram": v2_ngram(items),
         "V3_balance": v3_balance(items, meta),
         "V4_dedup_on_circuits": v4_dedup(items),
         "V5_split_integrity": v5_split_integrity(items, meta),
         "V6_token_identity_probe": v6_token_probe(items, meta),
         "V7_label_definition": v7_label_definition(items, meta),
         "V8_query_resolvability": v8_query_resolvability(items, meta)}
    r["gates"] = {"V1": r["V1_bag_of_embeddings"]["passed"],
                  "V3": r["V3_balance"]["passed"],
                  "V4": r["V4_dedup_on_circuits"]["passed"],
                  "V5": r["V5_split_integrity"]["passed"],
                  "V6": r["V6_token_identity_probe"]["passed"],
                  "V7": r["V7_label_definition"]["passed"],
                  "V8": r["V8_query_resolvability"]["passed"]}
    r["ungated_recorded"] = ["V2 (both halves, both T)",
                             "V6 unseen-pairings half (both T)"]
    r["kill_gate_passed"] = bool(all(r["gates"].values()))
    return r


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="doc 55 SS4.5: the CONFIRM draw uses seed + 1000 = %d, "
                         "registered in the doc, never chosen after the fact"
                         % CONFIRM_SEED)
    ap.add_argument("--out-a", default=OUT_A)
    ap.add_argument("--out-b", default=OUT_B)
    ap.add_argument("--validate", action="store_true",
                    help="run doc 55 SS7.5's V-suite kill-gate on both parts")
    ap.add_argument("--validation-out", default=VALIDATION_OUT)
    a = ap.parse_args()

    if a.validate:
        rep = {"experiment": "exp69 V-suite kill-gate (doc 55 SS7.5)",
               "spec": SPEC,
               "rederivation_note": (
                   "RE-DERIVED AT MULTI-EVENT SCALE, NOT INHERITED -- doc 55 SS7.5, "
                   "doc 50 SS5.1's first requirement and doc 46 SS8's lesson that "
                   "assuming a V-suite transfers is the doc-42 failure class. V1 is "
                   "RE-MEASURED at both T with twin features checked BIT-identical; "
                   "V2 is re-derived and reported UN-GATED with its reason; V3 adds "
                   "verb-pair, per-event surface order and query index; V4 dedups on "
                   "the full ORDERED event list; V5 adds the T1/T2 shared-split "
                   "invariant; V8 is NEW."),
               "parts": {}}
        rc = 0
        for part, path in (("A", a.out_a), ("B", a.out_b)):
            r = validate(path)
            rep["parts"][part] = r
            g = r["gates"]
            v1, v3, v4, v5, v8 = (r["V1_bag_of_embeddings"], r["V3_balance"],
                                  r["V4_dedup_on_circuits"], r["V5_split_integrity"],
                                  r["V8_query_resolvability"])
            print("\n[exp69 V-suite] PART %s  %s  sha %s"
                  % (part, path, r["dataset_sha256"][:12]))
            for t in (1, 2):
                c = v1["T%d" % t]
                print("  V1  T=%d bag-of-embeddings %.6f  twins-bit-identical=%s "
                      "(%d pairs)  %s"
                      % (t, c["test_accuracy"], c["twin_features_bit_identical"],
                         c["n_twin_pairs_checked"], "PASS" if c["passed"] else "FAIL"))
            for t in (1, 2):
                c = r["V2_ngram"]["T%d" % t]
                print("  V2  T=%d n-gram seen-pairings %.4f | unseen-pairings %.4f "
                      "  [RECORDED, NO BAR -- pairing split]"
                      % (t, c["seen_pairings_val_accuracy"],
                         c["unseen_pairings_test_accuracy"]))
            print("  V3  balance min_p %.4f  %s" % (v3["min_p"],
                                                    "PASS" if g["V3"] else "FAIL"))
            for t in (1, 2):
                c = v4["T%d" % t]
                print("  V4  T=%d %d distinct circuits of %d items, %d contradictory"
                      "  %s" % (t, c["n_distinct_circuits"], c["n_items"],
                                c["n_contradictory"], "PASS" if c["passed"] else "FAIL"))
            print("  V5  casts %d/%d/%d disjoint=%s coverage B=%s N=%s shared-split=%s"
                  "  n_test=%s  %s"
                  % (v5["n_train_casts"], v5["n_val_casts"], v5["n_test_casts"],
                     v5["casts_disjoint_across_splits"], v5["bearers_covered_in_train"],
                     v5["nonbearers_covered_in_train"], v5["T1_and_T2_share_one_split"],
                     v5["n_test_items"], "PASS" if g["V5"] else "FAIL"))
            for t in (1, 2):
                c = r["V6_token_identity_probe"]["T%d" % t]
                print("  V6  T=%d token probe seen %.4f (must>=%.2f) | unseen %.4f "
                      "[NO BAR]" % (t, c["seen_pairings_val_accuracy"], V6_TRAIN_BAR,
                                    c["unseen_pairings_test_accuracy"]))
            print("  V7  label violations %d, cast violations %d, balanced=%s  %s"
                  % (r["V7_label_definition"]["n_violations"],
                     r["V7_label_definition"]["n_cast_violations"],
                     r["V7_label_definition"]["balanced_in_every_split"],
                     "PASS" if g["V7"] else "FAIL"))
            print("  V8  queried-verb non-unique %d, wrong-event %d, slot violations "
                  "%d, slots disjoint=%s  %s"
                  % (v8["n_queried_verb_not_unique"], v8["n_queried_verb_in_wrong_event"],
                     v8["n_slot_violations"], v8["slots_disjoint"],
                     "PASS" if g["V8"] else "FAIL"))
            print("  KILL-GATE PART %s: %s"
                  % (part, "PASSED" if r["kill_gate_passed"] else "FAILED"))
            if not r["kill_gate_passed"]:
                rc = 1
        rep["kill_gate_passed"] = bool(all(rep["parts"][p]["kill_gate_passed"]
                                           for p in ("A", "B")))
        rep["consequence_if_failed"] = (
            "doc 46 SS9 / doc 55 SS7.5: any gate red and NO TRAINING RUNS -- a new "
            "doc is required, not a patch.")
        json.dump(rep, open(a.validation_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[exp69 V-suite] wrote %s -- KILL-GATE %s"
              % (a.validation_out, "PASSED" if rep["kill_gate_passed"] else "FAILED"))
        return rc

    if not os.path.exists(PREDICTIONS):
        raise RuntimeError(
            "%s does not exist. doc 55 SS19.2 / SS19.7 register the predictions file "
            "as coming BEFORE the dataset, and that ordering is what makes exp69 a "
            "PROSPECTIVE test of the calculus. Run exp69_predictions.py first."
            % PREDICTIONS)

    for part, out in (("A", a.out_a), ("B", a.out_b)):
        items, meta = build(part, a.seed)
        meta["predictions_file"] = PREDICTIONS
        meta["predictions_sha256"] = comp.canonical_sha256(
            json.load(open(PREDICTIONS, encoding="utf-8")))
        json.dump({"meta": meta, "items": items},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("[exp69] wrote %s  (part %s)" % (out, part))
        print("  category: %s (%s) -- %s"
              % (meta["category"]["key"], meta["category"]["gloss"],
                 " ".join(meta["bearers"])))
        print("  verb slots: V_A = %s | V_B = %s"
              % (" ".join(meta["verb_slots"]["V_A"]),
                 " ".join(meta["verb_slots"]["V_B"])))
        print("  casts: %d total = %d train / %d val / %d test  (pool draws %d, "
              "split draws %d)"
              % (meta["n_casts"], len(meta["train_casts"]), len(meta["val_casts"]),
                 len(meta["test_casts"]), meta["cast_pool_draws"],
                 meta["coverage_assertion"]["draws_until_satisfied"]))
        print("  splits: T=1 %s | T=2 %s"
              % (meta["split_counts"]["1"], meta["split_counts"]["2"]))
        print("  n_items: %d   sha256: %s" % (meta["n_items"], meta["dataset_sha256"]))
        ex1 = [i for i in items if i["T"] == 1][0]
        ex2 = [i for i in items if i["T"] == 2][0]
        print("  T=1 example: %s | q: %s | %s"
              % (" / ".join(ex1["sentences"]), ex1["question"]["surface"],
                 ex1["answer"]))
        print("  T=2 example (q=%d, %s): %s | q: %s | %s"
              % (ex2["query_index"], "/".join(ex2["order_flags"]),
                 " / ".join(ex2["sentences"]), ex2["question"]["surface"],
                 ex2["answer"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
