# -*- coding: utf-8 -*-
"""exp51_datagen.py -- doc 35, the verb-twin rung (VT).

exp49 showed L1 and L2 are solvable VERB-BLIND: a model with one shared
block for all six verbs matches on L1 and BEATS the verb-aware model on L2.
The mechanism is structural -- both are ARGUMENT-exchange twins, so
answering needs a block sensitive to argument order, not knowledge of which
verb. This builds the rung the ladder was missing.

CONSTRUCTION (doc 35 SS2): twins differ only in WHICH VERB attaches to WHICH
argument pair.

    twin YES   events: (v1, A, X), (v2, C, D), <fillers>
    twin NO    events: (v2, A, X), (v1, C, D), <fillers>
    question (both): "did A v1 X?"   -> YES for the first, NO for the second

with A, X, C, D four distinct referents and v1 != v2, the two core events at
the SAME slots in both twins, and identical fillers, cast and question.

WHY IT DOES NOT LEAK: the twins carry the same multiset of words -- same two
verbs, same four names -- differing only in the pairing, so V1/V2 cannot
separate them. The naive design (twin NO substitutes a different verb) would
let a bag model win by lexical matching and is explicitly REJECTED by doc 35.

THE TIED-BLOCK THEOREM (doc 35 SS3): with one shared block U, both twins apply
U on (A,X) and U on (C,D) at the same slots in the same order with identical
intros and fillers -- literally the same operation sequence -- so they compile
to the IDENTICAL statevector and any readout gives identical predictions.
Opposite labels then pin the tied model to EXACTLY 0.500. Measured while
drafting doc 35: max statevector delta in complex64 = exactly 0.000e+00.

Registered deviations (doc 35 SS9):
  * ALL-HUMAN casts, objects: [] for all six verbs. Required so that the
    verb-relabeling symmetry is exact: with standard casts, حمل/رفع filler
    events take object patients, and relabeling حمل->ضرب would produce
    out-of-distribution ضرب+object events. Uses the exp46 SS3 precedent.
    Cost: the VT distribution diverges from L1's, so cross-rung comparison
    is qualitative.
  * flip_idx is a LIST of the two exchanged slots (verb exchange flips two
    events; the exp41 schema carries one index). Checked before writing:
    flip_idx is consumed only by the generator's own twin construction and
    exp43a_analysis.py, NOT by the V-suite -- canonical_key keys on events,
    question and answer.

Fillers use only the four verbs OUTSIDE {v1, v2}, so no filler can ever
create the queried triple in the NO twin. Which pair is queried, and hence
which verb the question names, is drawn uniformly per item (doc 35 SS2).
"""

import hashlib
import json
import random
from collections import Counter

import exp41_story_generator as gen

DESIGN_DOC = ("qnlp_private_docs/35_exp51_verb_twin_rung.md "
              "(commit 6101ffe, FROZEN)")
RUNG = "VT"
SEED = 51000                                    # FROZEN (doc 35 SS11)
K, T = 4, 5                                     # FROZEN
SPLIT_PAIRS = {"train": 600, "val": 150, "test": 150}   # pairs -> 2x items
VERBS = ["حمل", "رفع", "سأل", "ساعد", "ضرب", "عالج"]    # the frozen L1 six


def canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_pair(rng, split, pair_no):
    cast = rng.sample(gen.HUMANS, K)            # all-human cast (doc 35 SS9)
    a, x, c, d = cast
    v1, v2 = rng.sample(VERBS, 2)
    others = [v for v in VERBS if v not in (v1, v2)]

    slots = sorted(rng.sample(range(T), 2))     # SAME slots in both twins
    s1, s2 = slots

    fillers = []
    used = set()
    while len(fillers) < T - 2:
        fv = rng.choice(others)                 # never v1/v2 -> cannot forge
        fa, fp = rng.sample(cast, 2)            # the queried triple
        key = (fv, fa, fp)
        if key in used:
            continue
        used.add(key)
        fillers = fillers + [{"verb": fv, "agent": fa, "patient": fp,
                              "pro_drop": False}]

    def assemble(core1, core2):
        ev, fi = [None] * T, iter(fillers)
        ev[s1], ev[s2] = core1, core2
        return [e if e is not None else next(fi) for e in ev]

    e_yes_1 = {"verb": v1, "agent": a, "patient": x, "pro_drop": False}
    e_yes_2 = {"verb": v2, "agent": c, "patient": d, "pro_drop": False}
    e_no_1 = {"verb": v2, "agent": a, "patient": x, "pro_drop": False}
    e_no_2 = {"verb": v1, "agent": c, "patient": d, "pro_drop": False}
    events_yes = assemble(e_yes_1, e_yes_2)
    events_no = assemble(e_no_1, e_no_2)

    # query one of the two pairs uniformly; the question names the verb that
    # sits on that pair in the YES twin (doc 35 SS2 randomisation)
    if rng.random() < 0.5:
        qv, qa, qp = v1, a, x
    else:
        qv, qa, qp = v2, c, d
    question = {"type": "Q0", "surface": f"هل {qv} {qa} {qp}",
                "v": qv, "a": qa, "p": qp}

    referents = [{"name": h, "type": "human"} for h in cast]
    out = []
    for events, ans in ((events_yes, "yes"), (events_no, "no")):
        sents, flags = gen.render(rng, events)
        out.append({"answer": ans, "events": events,
                    "sentences": sents, "order_flags": flags})
    rng.shuffle(out)
    pid = "p%05d" % pair_no
    return [{
        "id": "%s_%s_%s" % (split, pid, "ab"[j]), "pair_id": pid,
        "split": split, "qtype": "QV", "hop_count": 1,
        "pro_drop_item": False, "order_dependent": False,
        "answer": p["answer"], "K": K, "T": T,
        "flip_idx": [s1, s2],                   # LIST: verb exchange flips two
        "referents": referents, "events": p["events"],
        "sentences": p["sentences"], "order_flags": p["order_flags"],
        "question": question,
    } for j, p in enumerate(out)]


def generate():
    rng = random.Random(SEED)
    items, n = [], 0
    for split, npairs in SPLIT_PAIRS.items():
        for _ in range(npairs):
            n += 1
            items.extend(build_pair(rng, split, n))

    # ---- load-bearing invariants, asserted rather than assumed
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], []).append(it)
    for pid, (p, q) in by_pair.items():
        bag_p = Counter((" . ".join(p["sentences"]) + " "
                         + p["question"]["surface"]).split())
        bag_q = Counter((" . ".join(q["sentences"]) + " "
                         + q["question"]["surface"]).split())
        assert bag_p == bag_q, "twin bag mismatch in %s" % pid
        assert {p["answer"], q["answer"]} == {"yes", "no"}, pid
        assert p["split"] == q["split"], pid
        # the answer must follow from the events, in both twins
        for it in (p, q):
            qq = it["question"]
            hit = any(e["verb"] == qq["v"] and e["agent"] == qq["a"]
                      and e["patient"] == qq["p"] for e in it["events"])
            assert hit == (it["answer"] == "yes"), \
                "label/events mismatch in %s" % it["id"]
        # twins must differ ONLY by the verb assignment on the two core slots
        s1, s2 = p["flip_idx"]
        for i in range(T):
            ep, eq = p["events"][i], q["events"][i]
            if i in (s1, s2):
                assert (ep["agent"], ep["patient"]) == (eq["agent"],
                                                        eq["patient"]), pid
                assert ep["verb"] != eq["verb"], pid
            else:
                assert ep == eq, "filler differs in %s slot %d" % (pid, i)

    sha = canonical_sha256(items)
    ev = Counter()
    for it in items:
        for e in it["events"]:
            ev[e["verb"]] += 1
    qv = Counter(it["question"]["v"] for it in items)
    meta = {
        "experiment": "exp51", "rung": RUNG, "design_doc": DESIGN_DOC,
        "seed": SEED, "dataset_sha256": sha, "n_items": len(items),
        "split_counts": dict(Counter(it["split"] for it in items)),
        "answer_balance": dict(Counter(it["answer"] for it in items)),
        "verb_inventory": {v: {"human_patients": True, "objects": [],
                               "swap": True} for v in VERBS},
        "event_verb_counts": dict(ev), "question_verb_counts": dict(qv),
        "humans": gen.HUMANS, "objects": [],
        "holdout_verb_argument_pairs": [],
        "construction": (
            "VERB-EXCHANGE twins: (v1,A,X),(v2,C,D) vs (v2,A,X),(v1,C,D) at "
            "the SAME slots, identical fillers/cast/question. Bag-identical. "
            "A tied-block model compiles both twins to the IDENTICAL "
            "statevector and is pinned at exactly 0.500 (doc 35 SS3)."),
        "registered_deviations": {
            "all_human_cast": ("objects:[] for all six verbs so the "
                               "verb-relabeling symmetry is exact; exp46 SS3 "
                               "precedent. Cost: distribution diverges from "
                               "L1, cross-rung comparison is qualitative."),
            "flip_idx_is_a_list": ("verb exchange flips TWO events; flip_idx "
                                   "is the list of both slots. Not consumed "
                                   "by the V-suite (canonical_key keys on "
                                   "events/question/answer)."),
        },
        "invariants_asserted": [
            "twin bag-of-words identity (incl. question surface)",
            "twin answers are exactly {yes, no}",
            "label follows from events in BOTH twins",
            "twins differ ONLY in the verb assignment on the two core slots",
            "fillers drawn from the four verbs outside {v1,v2}, so no filler "
            "can forge the queried triple",
        ],
        "no_sign_off_gate": (
            "doc 35 SS10: all six verbs come from the frozen validated exp41 "
            "inventory and all carry human_patients=True; the construction "
            "only pairs them with human agents and patients. No new lexical "
            "judgment is made, so no native-speaker gate is required."),
    }
    out = "stories_exp51_%s.json" % RUNG
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False)
    print("[exp51] wrote %s" % out)
    print("[exp51] n=%d splits=%s balance=%s"
          % (len(items), meta["split_counts"], meta["answer_balance"]))
    print("[exp51] sha256=%s" % sha)
    print("[exp51] event verbs: min=%d max=%d over %d"
          % (min(ev.values()), max(ev.values()), len(ev)))
    print("[exp51] question verbs: min=%d max=%d over %d"
          % (min(qv.values()), max(qv.values()), len(qv)))
    print("[exp51] all twin invariants asserted OK")
    return meta


if __name__ == "__main__":
    generate()
