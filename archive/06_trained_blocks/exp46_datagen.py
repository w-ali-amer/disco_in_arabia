# -*- coding: utf-8 -*-
"""exp46_datagen.py -- doc 29, wazn Stage B dataset (rung W1).

Twelve verbs, four per morphological pattern (wazn), ALL transitive with a
human direct object. Holding the selectional profile CONSTANT BY
CONSTRUCTION is the design's whole point: doc 26 SS4.2 records that wazn
cannot be isolated from meaning, but argument structure CAN be removed as a
confound, and Form III verbs are overwhelmingly human-directed. With every
verb sharing human_patients=True / objects=[] / swap=True, selectional
profile cannot contribute to any block distance.

Inventory SIGNED OFF by the native-speaker reviewer 2026-08-13 (doc 29 SS3
gate), no substitutions. Each verb was confirmed (i) unambiguously of its
stated wazn and (ii) transitive with a DIRECT human object taking no
preposition -- a verb governing على / مع / إلى / بـ would break the
twin-flip, since the argument-exchanged sentence would not be grammatical.

Rung W1 = an L1-style atomic-swap rung at the new inventory. Named W1 so it
is never confused with the frozen L1 (6 verbs), which is untouched.

REGISTERED GENERATOR DEVIATION (doc 29 SS3), exactly one, required by the
design rather than chosen for convenience:
  exp41_story_generator.cast_story draws kh = randint(3, K-1) humans, i.e.
  it FORCES at least one object referent into every cast. No verb in this
  inventory can take an object patient, so that referent can never appear
  in an event and the generator's cast-coverage requirement fails --
  measured at 299/300 attempts before this override existed. cast_story is
  therefore overridden to draw an ALL-HUMAN cast (K humans, 0 objects),
  changing nothing else. This follows the exp43b_datagen precedent, which
  already monkeypatches gen.VERBS / gen.SWAP_VERBS / gen.SPLIT_SPEC at
  generation time and restores them in a finally block.

CAPACITY NOTE (doc 29 SS3): make_holdout draws
HOLDOUT_NAMES_PER_VERB * len(SWAP_VERBS) = 2 * 12 = 24 distinct names from
exactly 24 HUMANS. Twelve swap-capable verbs is the MAXIMUM this machinery
supports without enlarging HUMANS or reducing HOLDOUT_NAMES_PER_VERB.

The frozen noun embeddings are reused VERBATIM -- all 24 human referents
are already covered, so no embedding is fitted and no embedding file is
touched. The frozen exp41/exp42/exp43b datasets are untouched.
"""

import hashlib
import json
from collections import Counter

import exp41_story_generator as gen

DESIGN_DOC = ("qnlp_private_docs/29_exp46_wazn_stage_b.md "
              "(commits 1ac3819 criteria + e9c69d0 sign-off, FROZEN)")
SIGN_OFF = {
    "date": "2026-08-13",
    "reviewer": "native-speaker reviewer (project owner)",
    "verdict": "all twelve verbs approved, no substitutions",
    "criteria": ("(i) unambiguously of the stated wazn; (ii) transitive "
                 "with a DIRECT human object, no preposition"),
}

# doc 29 SS3, FROZEN after sign-off
WAZN = {
    "I":   ["ضرب", "سأل", "شكر", "خدم"],
    "II":  ["علّم", "كلّم", "ودّع", "ذكّر"],
    "III": ["ساعد", "قابل", "عالج", "راسل"],
}
ROOTS = {
    "ضرب": "ض.ر.ب", "سأل": "س.أ.ل", "شكر": "ش.ك.ر", "خدم": "خ.د.م",
    "علّم": "ع.ل.م", "كلّم": "ك.ل.م", "ودّع": "و.د.ع", "ذكّر": "ذ.ك.ر",
    "ساعد": "س.ع.د", "قابل": "ق.ب.ل", "عالج": "ع.ل.ج", "راسل": "ر.س.ل",
}
VERBS = [v for p in ("I", "II", "III") for v in WAZN[p]]
PATTERN_OF = {v: p for p, vs in WAZN.items() for v in vs}
SEED = 29000                                   # FROZEN (doc 29 SS3)
RUNG = "W1"
SPEC = {                                       # L1 sizes; Q0 = 1-hop swap
    "train": {"Q1": 0, "Q2": 0, "Q0": 600, "K": (4, 4), "T": (5, 5)},
    "val":   {"Q1": 0, "Q2": 0, "Q0": 150, "K": (4, 4), "T": (5, 5)},
    "test":  {"Q1": 0, "Q2": 0, "Q0": 150, "K": (4, 4), "T": (5, 5)},
}


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cast_all_human(rng, K, force_humans=()):
    """doc 29 SS3 registered deviation. Upstream cast_story draws
    kh = randint(3, K-1), forcing >=1 object referent; with an
    all-human-patient inventory that referent can never participate and
    cast coverage fails. Same signature, same sampling discipline."""
    humans = list(force_humans)
    remaining = [h for h in gen.HUMANS if h not in humans]
    humans += rng.sample(remaining, K - len(humans))
    return humans, []


def generate():
    inventory = {v: {"human_patients": True, "objects": [], "swap": True}
                 for v in VERBS}
    saved = (gen.VERBS, gen.SWAP_VERBS, gen.SPLIT_SPEC, gen.cast_story)
    try:
        gen.VERBS = dict(inventory)
        gen.SWAP_VERBS = [v for v, c in gen.VERBS.items() if c["swap"]]
        gen.SPLIT_SPEC = dict(SPEC)
        gen.cast_story = cast_all_human
        assert len(gen.SWAP_VERBS) * gen.HOLDOUT_NAMES_PER_VERB \
            <= len(gen.HUMANS), \
            ("doc 29 SS3 capacity note: %d swap verbs x %d holdout names "
             "exceeds %d HUMANS"
             % (len(gen.SWAP_VERBS), gen.HOLDOUT_NAMES_PER_VERB,
                len(gen.HUMANS)))
        items, holdout = gen.generate(SEED)
    finally:
        gen.VERBS, gen.SWAP_VERBS, gen.SPLIT_SPEC, gen.cast_story = saved

    sha = canonical_sha256(items)
    ev = Counter()
    for it in items:
        for e in it["events"]:
            ev[e["verb"]] += 1
    qv = Counter(it["question"]["v"] for it in items)
    meta = {
        "experiment": "exp46",
        "rung": RUNG,
        "design_doc": DESIGN_DOC,
        "sign_off": SIGN_OFF,
        "seed": SEED,
        "dataset_sha256": sha,
        "n_items": len(items),
        "split_counts": dict(Counter(it["split"] for it in items)),
        "answer_balance": dict(Counter(it["answer"] for it in items)),
        "wazn_inventory": {v: {"wazn": PATTERN_OF[v], "root": ROOTS[v],
                               "human_patients": True, "objects": [],
                               "swap": True} for v in VERBS},
        # standard key the V-suite and downstream tooling expect
        "verb_inventory": {v: dict(inventory[v]) for v in VERBS},
        "wazn_groups": WAZN,
        "event_verb_counts": dict(ev),
        "question_verb_counts": dict(qv),
        "humans": gen.HUMANS,
        "objects": [],
        "holdout_verb_argument_pairs": sorted([list(p) for p in holdout]),
        "registered_deviation": {
            "what": "cast_story overridden to an ALL-HUMAN cast",
            "why": ("upstream cast_story draws kh = randint(3, K-1), forcing "
                    ">=1 object referent; no verb in this inventory takes an "
                    "object patient, so that referent can never appear in an "
                    "event and cast coverage fails (measured 299/300 attempts "
                    "before the override). Required by the fixed-selectional-"
                    "profile design, not chosen for convenience."),
            "precedent": ("exp43b_datagen already monkeypatches gen.VERBS / "
                          "gen.SWAP_VERBS / gen.SPLIT_SPEC and restores them "
                          "in a finally block"),
            "authorised_by": "doc 29 SS3",
        },
        "notes": [
            "selectional profile is CONSTANT BY CONSTRUCTION: every verb is "
            "human_patients=True, objects=[], swap=True, so profile cannot "
            "contribute to any block distance (doc 29 SS2)",
            "12 swap verbs is the MAXIMUM make_holdout supports: "
            "2 names/verb x 12 = 24 = len(HUMANS), zero slack",
            "frozen noun embeddings reused verbatim; no embedding fitted",
            "frozen exp41/exp42/exp43b datasets untouched",
        ],
    }
    out = "stories_exp46_%s.json" % RUNG
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False)
    print("[exp46] wrote %s" % out)
    print("[exp46] n=%d splits=%s balance=%s"
          % (len(items), meta["split_counts"], meta["answer_balance"]))
    print("[exp46] sha256=%s" % sha)
    print("[exp46] event verb counts: min=%d max=%d over %d verbs"
          % (min(ev.values()), max(ev.values()), len(ev)))
    print("[exp46] question verb counts: min=%d max=%d over %d verbs"
          % (min(qv.values()), max(qv.values()), len(qv)))
    missing = [v for v in VERBS if v not in ev]
    assert not missing, "verbs absent from events: %s" % missing
    return meta


if __name__ == "__main__":
    generate()
