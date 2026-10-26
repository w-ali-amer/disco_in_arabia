#!/usr/bin/env python3
"""
exp41 — Adversarial Arabic story-QA generator (Phase 2, pre-registered design doc 22 §2).

Generates multi-hop Arabic story question-answering data for trained-block DisCoCirc
experiments (exp42). Every item ships with an argument-exchanged swap twin whose
answer flips and whose bag of words (story + question) is IDENTICAL — any bag-level
model is at chance BY CONSTRUCTION, not by statistical accident. Twins share a
pair_id and always land in the same split (exp21 leakage lesson).

Structural leak kills (adversarial design notes):
  * All human referents are masculine — verb agreement would otherwise change the
    token (ضرب/ضربت) under agent/patient exchange and break twin bag-identity.
  * "No" answers exist ONLY via twin argument exchange, never via a different
    referent/verb draw — so class-conditional marginals of every token are exactly
    symmetric (V3 is exact, not approximate).
  * Question templates are fixed per question type and identical across classes
    (class-symmetric surface); yes/no differ only in the gold structure.
  * Multi-hop by construction: Q1 (2-hop referent indirection through a relative
    clause) and Q2 (pro-drop agent recoverable only from discourse continuity).
    Interference constraints make the queried composition unique in the story.
  * Single-hop Q0 items are generated as a LABELED control class (train/val only,
    excluded from test by split assignment).
  * SVO/VSO per-sentence coin flip makes (verb, noun) bigrams role-ambiguous.

Sizes (doc 22 §5): N_train=2000, N_val=500, N_test=540 across three compositional
splits: (a) longer T=12–20, (b) larger K=10–12, (c) unseen verb–argument pairings
(held-out (verb, referent) co-occurrences used only in test_c core events).

Deterministic given --seed. Dataset SHA256 recorded in output meta and manifest.
Outputs: stories_exp41.json, stories_exp41_manifest.json.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter

# ----------------------------------------------------------------------------
# Vocabulary — harvested from sentences.json surface forms, EXP23 Tier-A root
# families, and the 6 audited verbs (ضرب حمل رفع قطع فتح + present forms).
# Humans are all masculine singular (see module docstring for why).
# ----------------------------------------------------------------------------

HUMANS = [
    "الطالب", "الرجل", "الولد", "المدير", "المعلم", "الطبيب", "العامل",
    "المهندس", "الشرطي", "الكاتب", "الطباخ", "الجندي", "الفلاح", "الحارس",
    "اللاعب", "الموظف", "الصحفي", "السائق", "المسافر", "الضابط", "الباحث",
    "القائد", "الصياد", "المريض",
]

OBJECTS = [
    "الكتاب", "الحقيبة", "الباب", "الرسالة", "الصندوق", "البيت", "الجدار",
    "السيارة", "الملف", "التقرير", "الورقة", "الخبز", "الصورة", "النافذة",
    "العقد", "القصيدة", "الوثيقة", "الدرس", "المكتب", "المشروع",
]

# verb -> {human_patients: bool, objects: [allowed object patients], swap: bool}
# swap=True verbs can take human agent AND human patient -> usable as the
# argument-exchanged (twin-flipped) event.
VERBS = {
    # audited verbs
    "ضرب": {"human_patients": True, "objects": [], "swap": True},
    "حمل": {"human_patients": True, "swap": True,
            "objects": ["الكتاب", "الحقيبة", "الصندوق", "الملف", "التقرير",
                        "الورقة", "الصورة", "الخبز"]},
    "رفع": {"human_patients": True, "swap": True,
            "objects": ["الكتاب", "الحقيبة", "الصندوق", "الملف", "التقرير",
                        "الورقة", "الصورة", "العقد"]},
    "قطع": {"human_patients": False, "swap": False,
            "objects": ["الورقة", "الخبز", "العقد"]},
    "فتح": {"human_patients": False, "swap": False,
            "objects": ["الباب", "النافذة", "الصندوق", "الملف", "الرسالة",
                        "الكتاب", "البيت", "المكتب"]},
    # Tier-A / Tier-B families
    "عالج": {"human_patients": True, "objects": [], "swap": True},   # ع.ل.ج
    "سأل": {"human_patients": True, "objects": [], "swap": True},    # س.أ.ل
    "ساعد": {"human_patients": True, "objects": [], "swap": True},   # س.ع.د
    "كتب": {"human_patients": False, "swap": False,                  # ك.ت.ب
            "objects": ["الرسالة", "التقرير", "القصيدة", "الدرس", "الوثيقة",
                        "المشروع"]},
    "درس": {"human_patients": False, "swap": False,                  # د.ر.س
            "objects": ["الكتاب", "التقرير", "المشروع", "الوثيقة", "القصيدة"]},
    "شرح": {"human_patients": False, "swap": False,                  # ش.ر.ح
            "objects": ["الدرس", "الكتاب", "التقرير", "المشروع", "القصيدة",
                        "الوثيقة"]},
    "كسر": {"human_patients": False, "swap": False,                  # ك.س.ر
            "objects": ["الباب", "النافذة", "الجدار", "الصندوق"]},
    "نظف": {"human_patients": False, "swap": False,                  # ن.ظ.ف
            "objects": ["البيت", "المكتب", "السيارة", "النافذة", "الجدار",
                        "الباب", "الصندوق"]},
    "رسم": {"human_patients": False, "swap": False,                  # ر.س.م
            "objects": ["الصورة", "البيت", "الجدار", "الباب", "السيارة"]},
    "فهم": {"human_patients": False, "swap": False,                  # ف.ه.م
            "objects": ["الدرس", "الكتاب", "التقرير", "القصيدة", "الرسالة",
                        "المشروع", "الوثيقة"]},
}

SWAP_VERBS = [v for v, c in VERBS.items() if c["swap"]]

# Split size spec: (qtype -> number of twin PAIRS); items = 2 * pairs.
SPLIT_SPEC = {
    "train":  {"Q1": 600, "Q2": 250, "Q0": 150, "K": (4, 8),   "T": (5, 10)},
    "val":    {"Q1": 150, "Q2": 65,  "Q0": 35,  "K": (4, 8),   "T": (5, 10)},
    "test_a": {"Q1": 68,  "Q2": 22,  "Q0": 0,   "K": (4, 8),   "T": (12, 20)},
    "test_b": {"Q1": 68,  "Q2": 22,  "Q0": 0,   "K": (10, 12), "T": (5, 10)},
    "test_c": {"Q1": 68,  "Q2": 22,  "Q0": 0,   "K": (4, 8),   "T": (5, 10)},
}

HOLDOUT_NAMES_PER_VERB = 2  # unseen verb-argument pairings for test_c


class StoryBuildError(Exception):
    pass


def pick(rng, seq):
    """rng.choice that raises StoryBuildError (-> retry) on an empty pool."""
    if not seq:
        raise StoryBuildError("empty pool")
    return rng.choice(seq)


def make_holdout(rng):
    """(verb, name) pairs banned from co-occurring in any event outside test_c
    core events. Ban is role-agnostic (agent OR patient) so the pairing is
    genuinely unseen for both members of a swap twin."""
    names = rng.sample(HUMANS, HOLDOUT_NAMES_PER_VERB * len(SWAP_VERBS))
    holdout = set()
    i = 0
    for v in SWAP_VERBS:
        for _ in range(HOLDOUT_NAMES_PER_VERB):
            holdout.add((v, names[i]))
            i += 1
    return holdout


def hits_holdout(holdout, verb, agent, patient):
    return (verb, agent) in holdout or (verb, patient) in holdout


def cast_story(rng, K, force_humans=()):
    kh = rng.randint(3, K - 1)  # >=3 humans, >=1 object
    kh = max(kh, len(force_humans), 3)
    humans = list(force_humans)
    remaining = [h for h in HUMANS if h not in humans]
    humans += rng.sample(remaining, kh - len(humans))
    objects = rng.sample(OBJECTS, K - kh)
    return humans, objects


def patient_pool(verb, cast_h, cast_o, exclude):
    pool = []
    if VERBS[verb]["human_patients"]:
        pool += [h for h in cast_h if h not in exclude]
    pool += [o for o in cast_o if o in VERBS[verb]["objects"] and o not in exclude]
    return pool


def sample_fillers(rng, n, cast_h, cast_o, core_events, banned_fn, holdout,
                   allow_holdout_core=False):
    """Sample n filler events. Prefers unused cast members so every referent
    appears in the story. Fillers never touch held-out (verb, name) pairs."""
    events = []
    seen = {(e["verb"], e["agent"], e["patient"]) for e in core_events}
    used = set()
    for e in core_events:
        used.add(e["agent"])
        used.add(e["patient"])

    for _ in range(n):
        placed = False
        for _try in range(400):
            unused = [r for r in cast_h + cast_o if r not in used]
            target = rng.choice(unused) if unused and rng.random() < 0.85 else None
            if target in cast_o:
                cands = [v for v in VERBS if target in VERBS[v]["objects"]]
                if not cands:
                    target = None
            verb = (rng.choice(cands) if target in cast_o
                    else rng.choice(list(VERBS)))
            if target in cast_h and rng.random() < 0.5 and VERBS[verb]["human_patients"]:
                agent = rng.choice([h for h in cast_h if h != target])
                patient = target
            else:
                agent = target if target in cast_h else rng.choice(cast_h)
                if target in cast_o:
                    patient = target
                else:
                    pool = patient_pool(verb, cast_h, cast_o, {agent})
                    if not pool:
                        continue
                    patient = rng.choice(pool)
            if agent == patient:
                continue
            ev = {"verb": verb, "agent": agent, "patient": patient,
                  "pro_drop": False}
            key = (verb, agent, patient)
            if key in seen:
                continue
            if hits_holdout(holdout, verb, agent, patient):
                continue
            if banned_fn(ev):
                continue
            events.append(ev)
            seen.add(key)
            used.add(agent)
            used.add(patient)
            placed = True
            break
        if not placed:
            raise StoryBuildError("filler sampling exhausted")

    unused = [r for r in cast_h + cast_o if r not in used]
    if unused:
        raise StoryBuildError("cast coverage failed")
    return events


def render(rng, events):
    sents, flags = [], []
    for e in events:
        if e["pro_drop"]:
            sents.append(f'{e["verb"]} {e["patient"]}')
            flags.append("PRO")
        elif rng.random() < 0.5:
            sents.append(f'{e["verb"]} {e["agent"]} {e["patient"]}')
            flags.append("VSO")
        else:
            sents.append(f'{e["agent"]} {e["verb"]} {e["patient"]}')
            flags.append("SVO")
    return sents, flags


def build_pair_core(rng, qtype, split, K_rng, T_rng, holdout):
    """Build one (yes-item, no-twin) pair. Returns dict with events for both
    twins (they differ in exactly one argument-exchanged event), the question,
    and metadata. Raises StoryBuildError to trigger a retry."""
    use_holdout_core = split == "test_c"
    K = rng.randint(*K_rng)
    # feasibility: T events expose <= 2T referent slots; ensure the cast fits
    tmin = max(T_rng[0], min((K + 5) // 2, T_rng[1]))
    T = rng.randint(tmin, T_rng[1])

    if qtype == "Q1":
        # e1 = (v1, A, X); e2 = (v2, A, Y); twin flips e2 -> (v2, Y, A).
        # Question: did the one who v1-ed X  v2  Y?
        v2 = rng.choice(SWAP_VERBS)
        if use_holdout_core:
            A = rng.choice([n for (v, n) in holdout if v == v2])
        else:
            A = rng.choice(HUMANS)
        humans, objects = cast_story(rng, K, force_humans=(A,))
        Y = pick(rng, [h for h in humans
                       if h != A and (v2, h) not in holdout])
        v1 = rng.choice([v for v in VERBS if v != v2])
        xpool = [x for x in patient_pool(v1, humans, objects, {A, Y})
                 if (v1, x) not in holdout]
        if not xpool or (v1, A) in holdout:
            raise StoryBuildError("Q1 pools empty / holdout clash")
        X = pick(rng, xpool)
        e1 = {"verb": v1, "agent": A, "patient": X, "pro_drop": False}
        e2 = {"verb": v2, "agent": A, "patient": Y, "pro_drop": False}
        if not use_holdout_core and hits_holdout(holdout, v2, A, Y):
            raise StoryBuildError("Q1 core hits holdout")

        def banned(ev):
            # uniqueness of the relative clause "the one who v1-ed X"
            if ev["verb"] == v1 and ev["patient"] == X:
                return True
            # nothing else may connect v2 with A or Y (answer uniqueness,
            # holds in both twins since exchange stays within {A, Y})
            if ev["verb"] == v2 and (ev["agent"] in (A, Y)
                                     or ev["patient"] in (A, Y)):
                return True
            return False

        fillers = sample_fillers(rng, T - 2, humans, objects, [e1, e2],
                                 banned, holdout)
        slots = [None] * T
        p1, p2 = rng.sample(range(T), 2)
        slots[p1], slots[p2] = e1, e2
        fit = iter(fillers)
        events = [s if s is not None else next(fit) for s in slots]
        flip_idx = events.index(e2)
        question = {"type": "Q1",
                    "surface": f"هل الذي {v1} {X} {v2} {Y}",
                    "v1": v1, "x": X, "v2": v2, "y": Y}
        meta = {"hop_count": 2, "pro_drop_item": False,
                "order_dependent": False}

    elif qtype == "Q2":
        # prev = (v3, A, C); drop = (v2, PRO=A, Y) — agent recoverable only
        # from discourse continuity (agent of the immediately preceding
        # sentence). Twin flips prev -> (v3, C, A) so PRO resolves to C.
        # Question: did A v2 Y?
        v3 = rng.choice(SWAP_VERBS)
        if use_holdout_core:
            A = rng.choice([n for (v, n) in holdout if v == v3])
        else:
            A = rng.choice(HUMANS)
        humans, objects = cast_story(rng, K, force_humans=(A,))
        C = pick(rng, [h for h in humans
                       if h != A and (v3, h) not in holdout])
        v2 = rng.choice([v for v in VERBS if v != v3])
        ypool = [y for y in patient_pool(v2, humans, objects, {A, C})
                 if (v2, y) not in holdout]
        if not ypool or (v2, A) in holdout or (v2, C) in holdout:
            raise StoryBuildError("Q2 pools empty / holdout clash")
        Y = pick(rng, ypool)
        prev = {"verb": v3, "agent": A, "patient": C, "pro_drop": False}
        drop = {"verb": v2, "agent": A, "patient": Y, "pro_drop": True}
        if not use_holdout_core and hits_holdout(holdout, v3, A, C):
            raise StoryBuildError("Q2 core hits holdout")

        def banned(ev):
            # nothing else may connect v2 with A, C or Y — the pro-dropped
            # event must stay the unique source of the answer in both twins
            if ev["verb"] == v2 and (ev["agent"] in (A, C, Y)
                                     or ev["patient"] in (A, C, Y)):
                return True
            return False

        fillers = sample_fillers(rng, T - 2, humans, objects, [prev, drop],
                                 banned, holdout)
        p = rng.randint(1, T - 2)  # drop sentence is 3rd or later
        events = []
        fit = iter(fillers)
        for i in range(T):
            if i == p:
                events.append(prev)
            elif i == p + 1:
                events.append(drop)
            else:
                events.append(next(fit))
        flip_idx = p  # twin exchanges prev's arguments
        question = {"type": "Q2",
                    "surface": f"هل {v2} {A} {Y}",
                    "v": v2, "a": A, "p": Y}
        meta = {"hop_count": 2, "pro_drop_item": True,
                "order_dependent": True}

    elif qtype == "Q0":
        # single-hop control: e = (v, A, X); question: did A v X?
        v = rng.choice(SWAP_VERBS)
        A = rng.choice([h for h in HUMANS if (v, h) not in holdout])
        humans, objects = cast_story(rng, K, force_humans=(A,))
        xs = [h for h in humans if h != A and (v, h) not in holdout]
        if not xs:
            raise StoryBuildError("Q0 pool empty")
        X = pick(rng, xs)
        e = {"verb": v, "agent": A, "patient": X, "pro_drop": False}

        def banned(ev):
            if ev["verb"] == v and (ev["agent"] in (A, X)
                                    or ev["patient"] in (A, X)):
                return True
            return False

        fillers = sample_fillers(rng, T - 1, humans, objects, [e],
                                 banned, holdout)
        slots = [None] * T
        slots[rng.randrange(T)] = e
        fit = iter(fillers)
        events = [s if s is not None else next(fit) for s in slots]
        flip_idx = events.index(e)
        question = {"type": "Q0",
                    "surface": f"هل {v} {A} {X}",
                    "v": v, "a": A, "p": X}
        meta = {"hop_count": 1, "pro_drop_item": False,
                "order_dependent": False}
    else:
        raise ValueError(qtype)

    twin_events = [dict(ev) for ev in events]
    fe = twin_events[flip_idx]
    fe["agent"], fe["patient"] = fe["patient"], fe["agent"]
    # re-resolve pro-drop gold agents by the discourse-continuity convention:
    # a dropped subject IS the previous sentence's agent, so exchanging the
    # previous event's arguments changes the dropped agent too (Q2 twins).
    for i, ev2 in enumerate(twin_events):
        if ev2["pro_drop"]:
            ev2["agent"] = twin_events[i - 1]["agent"]

    referents = ([{"name": h, "type": "human"} for h in humans]
                 + [{"name": o, "type": "object"} for o in objects])
    return {"events_yes": events, "events_no": twin_events,
            "flip_idx": flip_idx, "question": question,
            "referents": referents, "K": K, "T": T, **meta}


def canonical_key(item):
    """Alpha-renamed structural form for near-dup detection (surface order
    flags excluded; names replaced by first-occurrence indices)."""
    order = []

    def idx(name):
        if name not in order:
            order.append(name)
        return f"R{order.index(name)}"

    ev = [(e["verb"], idx(e["agent"]), idx(e["patient"]), e["pro_drop"])
          for e in item["events"]]
    q = item["question"]
    qc = [q["type"]] + [idx(t) if t in order or t in HUMANS + OBJECTS else t
                        for k, t in sorted(q.items()) if k not in ("type", "surface")]
    return json.dumps({"ev": ev, "q": qc, "a": item["answer"]},
                      ensure_ascii=False, sort_keys=True)


def generate(seed):
    rng = random.Random(seed)
    holdout = make_holdout(rng)
    items = []
    verbatim = set()
    pair_counter = 0

    for split, spec in SPLIT_SPEC.items():
        split_items = []
        for qtype in ("Q1", "Q2", "Q0"):
            for _ in range(spec[qtype]):
                for _attempt in range(300):
                    try:
                        core = build_pair_core(rng, qtype, split,
                                               spec["K"], spec["T"], holdout)
                    except StoryBuildError:
                        continue
                    pair = []
                    for answer, ekey in (("yes", "events_yes"),
                                         ("no", "events_no")):
                        events = core[ekey]
                        sents, flags = render(rng, events)
                        vkey = " . ".join(sents) + " || " + core["question"]["surface"]
                        pair.append({
                            "answer": answer, "events": events,
                            "sentences": sents, "order_flags": flags,
                            "vkey": vkey,
                        })
                    if any(p["vkey"] in verbatim for p in pair):
                        continue
                    if pair[0]["vkey"] == pair[1]["vkey"]:
                        continue  # degenerate render collision
                    pair_counter += 1
                    pid = f"p{pair_counter:05d}"
                    # random storage order within the pair
                    rng.shuffle(pair)
                    for j, p in enumerate(pair):
                        verbatim.add(p.pop("vkey"))
                        split_items.append({
                            "id": f"{split}_{pid}_{'ab'[j]}",
                            "pair_id": pid,
                            "split": split,
                            "qtype": qtype,
                            "hop_count": core["hop_count"],
                            "pro_drop_item": core["pro_drop_item"],
                            "order_dependent": core["order_dependent"],
                            "answer": p["answer"],
                            "K": core["K"], "T": core["T"],
                            "flip_idx": core["flip_idx"],
                            "referents": core["referents"],
                            "events": p["events"],
                            "sentences": p["sentences"],
                            "order_flags": p["order_flags"],
                            "question": core["question"],
                        })
                    break
                else:
                    raise RuntimeError(
                        f"could not build pair {qtype}/{split} in 300 attempts")
        rng.shuffle(split_items)
        items.extend(split_items)

    # sanity: twin bag-of-words identity (the load-bearing invariant)
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], []).append(it)
    for pid, (a, b) in by_pair.items():
        bag_a = Counter((" . ".join(a["sentences"]) + " "
                         + a["question"]["surface"]).split())
        bag_b = Counter((" . ".join(b["sentences"]) + " "
                         + b["question"]["surface"]).split())
        assert bag_a == bag_b, f"twin bag mismatch in {pid}"
        assert {a["answer"], b["answer"]} == {"yes", "no"}, pid
        assert a["split"] == b["split"], f"twin split leak in {pid}"

    return items, holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default="stories_exp41.json")
    ap.add_argument("--manifest", default="stories_exp41_manifest.json")
    args = ap.parse_args()

    items, holdout = generate(args.seed)
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    counts = Counter((it["split"], it["qtype"]) for it in items)
    split_counts = Counter(it["split"] for it in items)
    meta = {
        "experiment": "exp41",
        "design_doc": "qnlp_private_docs/22_phase2_trained_story_qa_design.md §2 (commit cacac31)",
        "seed": args.seed,
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
            "every item has a bag-of-words-identical swap twin with flipped answer (same split)",
            "all human referents masculine so argument exchange preserves the token bag",
            "Q0 single-hop items are the labeled control class, train/val only",
            "test_c core events use held-out (verb, referent) pairings unseen in any other split",
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False, indent=1)

    manifest = {k: v for k, v in meta.items() if k != "verb_inventory"}
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"[exp41] wrote {args.out}: {len(items)} items")
    for s in SPLIT_SPEC:
        print(f"  {s:8s} {split_counts[s]:5d}  "
              + "  ".join(f"{q}={counts[(s, q)]}" for q in ("Q1", "Q2", "Q0")))
    print(f"[exp41] dataset sha256 = {sha}")


if __name__ == "__main__":
    main()
