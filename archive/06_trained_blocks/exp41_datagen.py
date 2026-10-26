#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp41_datagen.py -- the REVIVAL story generator (doc 22 SS2/SS5, doc 65 SS2.3).

REGISTRATION
    design of record : qnlp_private_docs/22_phase2_trained_story_qa_design.md SS2, SS5
    amended by       : qnlp_private_docs/65_exp40_42_revival.md SS2.3 (hygiene:
                       designed-to-fail smoke builds), SS2.6 (Mac env)
    validator        : validate_exp41_data.py (V1-V5, doc 22 SS2.3 verbatim)

RELATION TO THE AUGUST BUILD (exp41_story_generator.py, seed 41, sha 2cb3cc..)
    That file is the COMMITTED VOCABULARY SOURCE and is imported here, never
    retyped and never modified: HUMANS / OBJECTS / VERBS / SWAP_VERBS, the
    3-word parser-safe SVO/VSO renderer, the cast/filler samplers and the
    held-out (verb, referent) machinery all come from it by import. exp69_datagen
    imports the same module (`import exp41_story_generator as gen`) and asserts
    its HUMANS against the D4-audited inventory in instrument_sb_bounds, so this
    inventory is the committed one of record.

    THREE THINGS ARE NEW HERE, and they are the reason this is a new file:

    (1) THE BALANCING EVENT -- the >= 2-hop fix (doc 22 SS2.2). The August Q1
        built the queried event so that NOTHING else in the story connected the
        query verb v2 with the queried patient Y. That uniqueness makes the item
        SINGLE-SENTENCE ANSWERABLE: "does some sentence read (v2, *, Y)?" is
        true exactly in the yes-twin, because the no-twin's flipped event reads
        (v2, Y, *). The August V5 oracle was a mean-pooled bag-of-embeddings
        model, blind to word order, so it could not see the shortcut; it scored
        exactly 0.500 and V5 passed. Under the order-aware, question-matching V5
        oracle of the revival validator that same construction FAILS -- measured,
        via the SAB_NODISTRACT smoke build, which IS the August construction.

        The fix: each item carries ONE EXTRA EVENT, and the twin exchanges the
        arguments of TWO events rather than one, chosen so that the multiset of
        QUESTION-RELATIVE SENTENCE PATTERNS is IDENTICAL in the two twins.

            Q1  yes:  (v1, A, X)   (v2, A, Y)   (v2, Y, Z)
                no :  (v1, A, X)   (v2, Y, A)   (v2, Z, Y)
            Q2  yes:  (v3, A, C)   (v2, PRO, Y)   (v4, C, A)
                no :  (v3, C, A)   (v2, PRO, Y)   (v4, A, C)

        In Q1 the question names v1, X, v2 and Y but NOT A or Z; both twins
        therefore contain exactly one sentence of pattern (verb = q.v2, q.Y as
        patient) and one of pattern (verb = q.v2, q.Y as agent). In Q2 the
        question names v2, A and Y but not v3, v4 or C; both twins contain
        exactly one sentence with q.A as agent and one with q.A as patient.
        Existence counts, and even per-pattern COUNTS, are matched -- so no
        function of a single sentence and the question separates the twins.
        What separates them is WHICH referent fills the unnamed slot, and that
        referent is tied to the question only through a second sentence. That
        is the >= 2-hop requirement made real rather than asserted. Fillers are
        byte-identical across a twin pair and so can only dilute, never leak.

    (2) SPLIT SIZES. Doc 22 SS5's N_test >= 500 is honoured PER SPLIT
        (test_a / test_b / test_c each 500 items) rather than on the union;
        the union is 1,500. train = 2,000 items, val = 500 items, as registered.

    (3) SEED. New registered constant SEED = 41000 (the August draw was 41),
        chosen so doc 65 SS2.2's confirm draw at SEED + 1000 = 42000 is free.

OUTPUTS (research rule 3: the August artefacts are NEVER overwritten)
    stories_exp41r.json, stories_exp41r_manifest.json

TASK DESIGN, in one place (doc 22 SS2)
    referents  K = 4-8 (train/val/test_a/test_c), 10-12 (test_b); humans are all
               masculine singular so an argument exchange preserves the token bag
    events     T = 5-10 (train/val/test_b/test_c), 12-20 (test_a); each event is
               rendered SVO or VSO by coin flip (same proposition, both orders)
    questions  Q1 two-hop referent indirection through a relative clause
               Q2 two-hop pro-drop: a later sentence drops its subject, which is
                  recoverable only by tracking the referent's wire
               Q0 single-hop LABELED CONTROL class, train/val only, never in test
               hop_count is stored per item; the question surface is fixed per
               question type and therefore class-symmetric by construction
    twins      every item ships an argument-exchanged twin with the flipped
               answer and an IDENTICAL bag of words; twins share a pair_id and
               are locked into the same split
    splits     (a) test_a longer stories, (b) test_b more referents,
               (c) test_c held-out (verb, referent) pairings in the core events

SMOKE BUILDS (doc 65 SS2.3) -- generated ONLY through --sabotage, written to
quarantined paths, aggregated in a SEPARATE block of the validation JSON, and
deleted afterwards. They must FAIL; a passing smoke build indicts the VALIDATOR.
    qleak       question particle differs by answer class      -> must fail V2
    singlehop   Q0 (single-sentence-answerable) items in test  -> must fail V5
    nodistract  the August construction, distractors removed   -> must fail V5

USAGE
    python exp41_datagen.py
    python exp41_datagen.py --seed 42000 --out stories_exp41r_confirm.json ...
    python exp41_datagen.py --sabotage qleak --out stories_exp41r_sab_qleak.json ...
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter

# The committed vocabulary + surface conventions. Imported, never retyped.
import exp41_story_generator as gen
from exp41_story_generator import (
    HUMANS, OBJECTS, VERBS, SWAP_VERBS,
    StoryBuildError, pick, cast_story, patient_pool, sample_fillers,
    make_holdout, hits_holdout, render,
)

SPEC = ("qnlp_private_docs/22_phase2_trained_story_qa_design.md SS2/SS5 "
        "+ 65_exp40_42_revival.md SS2.3/SS2.6")

SEED = 41000                     # registered generator seed for the revival
CONFIRM_SEED = SEED + 1000       # doc 65 SS2.2's confirm draw
OUT = "stories_exp41r.json"
MANIFEST = "stories_exp41r_manifest.json"

# pairs per (split, qtype); items = 2 * pairs (a pair is one item + its twin)
SPLIT_SPEC = {
    "train":  {"Q1": 600, "Q2": 250, "Q0": 150, "K": (4, 8),   "T": (5, 10)},
    "val":    {"Q1": 150, "Q2": 65,  "Q0": 35,  "K": (4, 8),   "T": (5, 10)},
    "test_a": {"Q1": 190, "Q2": 60,  "Q0": 0,   "K": (4, 8),   "T": (12, 20)},
    "test_b": {"Q1": 190, "Q2": 60,  "Q0": 0,   "K": (10, 12), "T": (5, 10)},
    "test_c": {"Q1": 190, "Q2": 60,  "Q0": 0,   "K": (4, 8),   "T": (5, 10)},
}

BUILD_ATTEMPTS = 3000
SABOTAGE_MODES = ("qleak", "singlehop", "nodistract")


# --------------------------------------------------------------------------
# question surfaces -- one fixed template per question type (class-symmetric).
# The `qleak` smoke build is the ONLY thing that ever varies them by class.
# --------------------------------------------------------------------------
Q_PARTICLE = "هل"
Q_PARTICLE_LEAK = "أ"            # a second, equally grammatical yes/no particle


def q_particle(answer, sabotage):
    if sabotage == "qleak":
        return Q_PARTICLE if answer == "yes" else Q_PARTICLE_LEAK
    return Q_PARTICLE


def q_surface(question, answer, sabotage):
    p = q_particle(answer, sabotage)
    t = question["type"]
    if t == "Q1":
        return f'{p} الذي {question["v1"]} {question["x"]} ' \
               f'{question["v2"]} {question["y"]}'
    return f'{p} {question["v"]} {question["a"]} {question["p"]}'


# --------------------------------------------------------------------------
# minimum story length: T must leave room for the core events AND for enough
# filler slots to mention every cast member (sample_fillers enforces coverage).
# --------------------------------------------------------------------------
def min_T(K, n_core, n_core_refs, T_rng):
    need = n_core + max(0, -(-(K - n_core_refs) // 2))
    return max(T_rng[0], min(need, T_rng[1]))


def three_distinct_swap_verbs(rng, exclude):
    pool = [v for v in SWAP_VERBS if v not in exclude]
    if len(pool) < 3:
        raise StoryBuildError("swap-verb pool too small")
    return rng.sample(pool, 3)


# --------------------------------------------------------------------------
def build_pair_core(rng, qtype, split, K_rng, T_rng, holdout, distractors=True):
    """One (yes, no) twin pair. The twins differ in exactly ONE event, whose
    arguments are exchanged; every other event is byte-identical, so the bag of
    words is identical and the answer flips. Raises StoryBuildError to retry."""
    use_holdout_core = split == "test_c"
    K = rng.randint(*K_rng)

    if qtype == "Q1":
        #  yes   e1 = (v1, A, X)   e2 = (v2, A, Y)   e3 = (v2, Y, Z)
        #  no    e1 = (v1, A, X)   e2 = (v2, Y, A)   e3 = (v2, Z, Y)
        #  ask   "did the one who v1-ed X  v2  Y?"       yes iff A v2-ed Y
        v2 = rng.choice(SWAP_VERBS)
        if use_holdout_core:
            # sorted(): `holdout` is a SET, and set iteration order over strings
            # varies between processes (PYTHONHASHSEED), which would make the
            # dataset sha irreproducible on a re-run. Determinism is checked by
            # generating twice in two processes and diffing the shas.
            A = rng.choice(sorted(n for (v, n) in holdout if v == v2))
        else:
            A = rng.choice(HUMANS)
        humans, objects = cast_story(rng, K, force_humans=(A,))
        cand = [h for h in humans if h != A and (v2, h) not in holdout]
        if len(cand) < 2:
            raise StoryBuildError("Q1 needs two spare humans for Y and Z")
        Y, Z = rng.sample(cand, 2)
        v1 = rng.choice([v for v in VERBS if v != v2])
        # X must differ from Z as well as from A and Y: if X == Z then the
        # balancing event reads (v2, q.Y, q.X) in the yes-twin and
        # (v2, q.X, q.Y) in the no-twin, and the ROLE OF q.X becomes a
        # single-sentence discriminator. (Measured leak: V5 strong-oracle
        # 0.598 with X == Z admitted in ~30% of Q1 items; see the report.)
        xpool = [x for x in patient_pool(v1, humans, objects, {A, Y, Z})
                 if (v1, x) not in holdout]
        if not xpool or (v1, A) in holdout:
            raise StoryBuildError("Q1 pools empty / holdout clash")
        X = pick(rng, xpool)
        if not use_holdout_core and hits_holdout(holdout, v2, A, Y):
            raise StoryBuildError("Q1 core hits holdout")

        e1 = {"verb": v1, "agent": A, "patient": X, "pro_drop": False}
        e2 = {"verb": v2, "agent": A, "patient": Y, "pro_drop": False}
        core = [e1, e2]
        if distractors:
            core.append({"verb": v2, "agent": Y, "patient": Z,
                         "pro_drop": False})

        def banned(ev):
            # "the one who v1-ed X" must denote exactly one referent
            if ev["verb"] == v1 and ev["patient"] == X:
                return True
            # the story's only v2 events are the core ones
            if ev["verb"] == v2:
                return True
            return False

        core_refs = {A, X, Y} | ({Z} if distractors else set())
        T = rng.randint(min_T(K, len(core), len(core_refs), T_rng), T_rng[1])
        fillers = sample_fillers(rng, T - len(core), humans, objects, core,
                                 banned, holdout)
        slots = [None] * T
        for p, ev in zip(rng.sample(range(T), len(core)), core):
            slots[p] = ev
        fit = iter(fillers)
        events = [s if s is not None else next(fit) for s in slots]
        flips = [(events.index(e2), (A, Y))]
        if distractors:
            flips.append((events.index(core[2]), (Y, Z)))
        question = {"type": "Q1", "v1": v1, "x": X, "v2": v2, "y": Y}
        meta = {"hop_count": 2, "pro_drop_item": False, "order_dependent": False}

    elif qtype == "Q2":
        #  yes   prev = (v3, A, C)   drop = (v2, PRO, Y)   bal = (v4, C, A)
        #  no    prev = (v3, C, A)   drop = (v2, PRO, Y)   bal = (v4, A, C)
        #  ask   "did A v2 Y?"     yes iff the dropped subject resolves to A,
        #  which it does iff A is the agent of the immediately preceding event.
        v3, v4, _ = three_distinct_swap_verbs(rng, exclude=())
        if use_holdout_core:
            A = rng.choice(sorted(n for (v, n) in holdout if v == v3))  # see Q1
        else:
            A = rng.choice(HUMANS)
        humans, objects = cast_story(rng, K, force_humans=(A,))
        C = pick(rng, [h for h in humans if h != A
                       and (v3, h) not in holdout and (v4, h) not in holdout])
        v2 = rng.choice([v for v in VERBS if v not in (v3, v4)])
        ypool = [y for y in patient_pool(v2, humans, objects, {A, C})
                 if (v2, y) not in holdout]
        if not ypool or (v2, A) in holdout or (v2, C) in holdout:
            raise StoryBuildError("Q2 pools empty / holdout clash")
        Y = pick(rng, ypool)
        if not use_holdout_core and hits_holdout(holdout, v3, A, C):
            raise StoryBuildError("Q2 core hits holdout")
        if (v4, A) in holdout:
            raise StoryBuildError("Q2 balancing event hits holdout")

        prev = {"verb": v3, "agent": A, "patient": C, "pro_drop": False}
        drop = {"verb": v2, "agent": A, "patient": Y, "pro_drop": True}
        bal = ({"verb": v4, "agent": C, "patient": A, "pro_drop": False}
               if distractors else None)

        def banned(ev):
            # the pro-dropped event must stay the story's only v2 event
            if ev["verb"] == v2:
                return True
            # no other v3/v4 event may touch A or C, so the role evidence for A
            # is exactly `prev` and `bal`
            if ev["verb"] in (v3, v4) and (ev["agent"] in (A, C)
                                           or ev["patient"] in (A, C)):
                return True
            return False

        core = [prev, drop] + ([bal] if bal else [])
        T = rng.randint(min_T(K, len(core), len({A, C, Y}), T_rng), T_rng[1])
        fillers = sample_fillers(rng, T - len(core), humans, objects, core,
                                 banned, holdout)
        # prev at p, drop at p+1 (the dropped subject is the previous agent);
        # `bal` and the fillers take the remaining slots in random order.
        p = rng.randint(1, T - 2)
        rest = ([bal] if bal else []) + fillers
        rng.shuffle(rest)
        events, fit = [], iter(rest)
        for i in range(T):
            events.append(prev if i == p else drop if i == p + 1 else next(fit))
        flips = [(p, (A, C))]
        if bal:
            flips.append((events.index(bal), (C, A)))
        question = {"type": "Q2", "v": v2, "a": A, "p": Y}
        meta = {"hop_count": 2, "pro_drop_item": True, "order_dependent": True}

    elif qtype == "Q0":
        # single-hop labelled control: one event, question names all three slots
        v = rng.choice(SWAP_VERBS)
        A = rng.choice([h for h in HUMANS if (v, h) not in holdout])
        humans, objects = cast_story(rng, K, force_humans=(A,))
        xs = [h for h in humans if h != A and (v, h) not in holdout]
        if not xs:
            raise StoryBuildError("Q0 pool empty")
        X = pick(rng, xs)
        e = {"verb": v, "agent": A, "patient": X, "pro_drop": False}

        def banned(ev):
            return ev["verb"] == v

        core = [e]
        T = rng.randint(min_T(K, 1, len({A, X}), T_rng), T_rng[1])
        fillers = sample_fillers(rng, T - 1, humans, objects, core,
                                 banned, holdout)
        slots = [None] * T
        slots[rng.randrange(T)] = e
        fit = iter(fillers)
        events = [s if s is not None else next(fit) for s in slots]
        flips = [(events.index(e), (A, X))]
        question = {"type": "Q0", "v": v, "a": A, "p": X}
        meta = {"hop_count": 1, "pro_drop_item": False, "order_dependent": False}
    else:
        raise ValueError(qtype)

    twin_events = [dict(ev) for ev in events]
    for idx, _pairnames in flips:
        fe = twin_events[idx]
        fe["agent"], fe["patient"] = fe["patient"], fe["agent"]
    # a dropped subject IS the previous sentence's agent, so exchanging the
    # previous event's arguments re-resolves it (this is the Q2 flip).
    for i, ev2 in enumerate(twin_events):
        if ev2["pro_drop"]:
            ev2["agent"] = twin_events[i - 1]["agent"]

    referents = ([{"name": h, "type": "human"} for h in humans]
                 + [{"name": o, "type": "object"} for o in objects])
    return {"events_yes": events, "events_no": twin_events,
            "flip_idxs": sorted(i for i, _ in flips),
            "question": question,
            "referents": referents, "K": K, "T": len(events), **meta}


# --------------------------------------------------------------------------
def generate(seed, sabotage=None, spec=None):
    """Deterministic given `seed`. `sabotage` is None or a SABOTAGE_MODES name."""
    if sabotage not in (None,) + SABOTAGE_MODES:
        raise ValueError(sabotage)
    spec = spec or SPLIT_SPEC
    distractors = sabotage != "nodistract"
    rng = random.Random(seed)
    holdout = make_holdout(rng)
    items, verbatim, pair_counter = [], set(), 0

    for split, sp in spec.items():
        split_items = []
        qtypes = ("Q1", "Q2", "Q0")
        if sabotage == "singlehop" and split.startswith("test_"):
            # the smoke build puts the single-hop control class INTO test
            sp = dict(sp, Q0=sp["Q1"] + sp["Q2"], Q1=0, Q2=0)
        for qtype in qtypes:
            for _ in range(sp[qtype]):
                for _attempt in range(BUILD_ATTEMPTS):
                    try:
                        core = build_pair_core(rng, qtype, split, sp["K"],
                                               sp["T"], holdout, distractors)
                    except StoryBuildError:
                        continue
                    pair = []
                    for answer, ekey in (("yes", "events_yes"),
                                         ("no", "events_no")):
                        events = core[ekey]
                        sents, flags = render(rng, events)
                        qs = q_surface(core["question"], answer, sabotage)
                        pair.append({"answer": answer, "events": events,
                                     "sentences": sents, "order_flags": flags,
                                     "qsurface": qs,
                                     "vkey": " . ".join(sents) + " || " + qs})
                    if any(p["vkey"] in verbatim for p in pair):
                        continue
                    if pair[0]["vkey"] == pair[1]["vkey"]:
                        continue                      # degenerate render collision
                    pair_counter += 1
                    pid = f"p{pair_counter:05d}"
                    rng.shuffle(pair)                 # random storage order
                    for j, p in enumerate(pair):
                        verbatim.add(p.pop("vkey"))
                        q = dict(core["question"])
                        q["surface"] = p["qsurface"]
                        split_items.append({
                            "id": f"{split}_{pid}_{'ab'[j]}",
                            "pair_id": pid, "split": split, "qtype": qtype,
                            "hop_count": core["hop_count"],
                            "pro_drop_item": core["pro_drop_item"],
                            "order_dependent": core["order_dependent"],
                            "answer": p["answer"],
                            "K": core["K"], "T": core["T"],
                            "flip_idxs": core["flip_idxs"],
                            "referents": core["referents"],
                            "events": p["events"],
                            "sentences": p["sentences"],
                            "order_flags": p["order_flags"],
                            "question": q,
                        })
                    break
                else:
                    raise RuntimeError(
                        f"could not build a {qtype} pair for {split} in "
                        f"{BUILD_ATTEMPTS} attempts")
        rng.shuffle(split_items)
        items.extend(split_items)

    check_invariants(items, sabotage)
    return items, holdout


def check_invariants(items, sabotage=None):
    """The load-bearing construction guarantees, asserted not assumed."""
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], []).append(it)
    for pid, twins in by_pair.items():
        assert len(twins) == 2, pid
        a, b = twins
        assert {a["answer"], b["answer"]} == {"yes", "no"}, pid
        assert a["split"] == b["split"], f"twin split leak in {pid}"
        if sabotage != "qleak":
            bags = [Counter((" . ".join(t["sentences"]) + " "
                             + t["question"]["surface"]).split()) for t in twins]
            assert bags[0] == bags[1], f"twin bag mismatch in {pid}"
        # the twins differ in exactly one event
        diff = [i for i, (x, y) in enumerate(zip(a["events"], b["events"]))
                if (x["agent"], x["patient"]) != (y["agent"], y["patient"])]
        assert 1 <= len(diff) <= 3, f"{pid}: {len(diff)} events differ"
        assert [e["verb"] for e in a["events"]] == [e["verb"] for e in b["events"]], \
            f"twin verb sequence differs in {pid}"
    if sabotage != "singlehop":
        assert all(it["hop_count"] >= 2
                   for it in items if it["split"].startswith("test_")), \
            "single-hop item leaked into test"


# --------------------------------------------------------------------------
def write(items, holdout, seed, out, manifest_path, sabotage=None):
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    counts = Counter((it["split"], it["qtype"]) for it in items)
    split_counts = Counter(it["split"] for it in items)
    meta = {
        "experiment": "exp41r" if sabotage is None else f"exp41r_SAB_{sabotage}",
        "design_doc": SPEC,
        "vocabulary_source": "exp41_story_generator.py (imported, not retyped)",
        "seed": seed,
        "sabotage": sabotage,
        "dataset_sha256": sha,
        "n_items": len(items),
        "split_counts": dict(split_counts),
        "split_qtype_counts": {f"{s}/{q}": n for (s, q), n in sorted(counts.items())},
        "verb_inventory": {v: {"human_patients": c["human_patients"],
                               "swap": c["swap"], "objects": c["objects"]}
                           for v, c in VERBS.items()},
        "humans": HUMANS, "objects": OBJECTS,
        "holdout_verb_argument_pairs": sorted([list(p) for p in holdout]),
        "notes": [
            "every item has a bag-of-words-identical swap twin with flipped "
            "answer, locked into the same split",
            "all human referents are masculine so an argument exchange "
            "preserves the token bag exactly",
            "Q1/Q2 carry SHARED role distractors so both question-relative "
            "sentence patterns occur in BOTH twins (the >=2-hop fix; the "
            "August build lacked them and is single-sentence answerable)",
            "Q0 single-hop items are the labelled control class, train/val only",
            "test_c core events use held-out (verb, referent) pairings unseen "
            "in any other split",
            "sentences are 3-word SVO/VSO frames (2 words when pro-dropped); "
            "gold structure is by construction, no parser is on the path",
        ],
    }
    if sabotage:
        meta["notes"].insert(0, "DESIGNED-TO-FAIL SMOKE BUILD (doc 65 SS2.3) -- "
                                "must fail its V-item; never trained on")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False, indent=1)
    if manifest_path:
        man = {k: v for k, v in meta.items() if k != "verb_inventory"}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
    return sha, split_counts, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--sabotage", choices=SABOTAGE_MODES, default=None)
    args = ap.parse_args()

    items, holdout = generate(args.seed, sabotage=args.sabotage)
    sha, split_counts, counts = write(items, holdout, args.seed, args.out,
                                      args.manifest, args.sabotage)
    tag = args.sabotage or "clean"
    print(f"[exp41r/{tag}] wrote {args.out}: {len(items)} items")
    for s in SPLIT_SPEC:
        print(f"  {s:8s} {split_counts[s]:5d}  "
              + "  ".join(f"{q}={counts[(s, q)]}" for q in ("Q1", "Q2", "Q0")))
    print(f"[exp41r/{tag}] dataset sha256 = {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
