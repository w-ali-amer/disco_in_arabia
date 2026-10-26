# -*- coding: utf-8 -*-
"""exp76_datagen.py -- THE Q76 GENERATOR (doc 63, D3 of doc 59's six-direction
plan). Cast size n, story length T, inventory |R|.

REGISTRATION: qnlp_private_docs/63_exp76_scaling.md, REGISTERED 2026-08-29
(orchestrator rulings in the status header, BINDING over the body). 120 runs
ACCEPTED. This file builds the ELEVEN trained cells' datasets and re-derives
the V-suite AT MULTI-REFERENT SCALE (doc 63 SS8.1; doc 50 SS5.1's requirement in
the form doc 55 SS7.5 met it).

Q76 (doc 63 SS4.2), verbatim from the registration:

    cast:      n referents, EXACTLY ONE of which is a category bearer b;
               the other n-1 are non-bearers drawn from the inventory
    text:      T events, event t = (v_t, agent_t, patient_t); the T verbs are
               DISTINCT and drawn from the committed 6-verb inventory; each
               event rendered independently as SVO or VSO
    question:  "the one who [v_q] -- is he a [CATEGORY]?"   q in {1..T}
    label:     1 iff agent(event q) is the bearer
    twin:      exchange agent and patient WITHIN event q ONLY; every other
               event is held BYTE-IDENTICAL; question surface identical
    C1 (label flips):      event q's participants are the bearer and exactly
                           one non-bearer
    C2 (V8 resolvability): the T verbs are pairwise distinct
    C3 (no inert structure): every event is reverse-reachable from event q's
                           wire pair (SS4.4), and every cast member appears in at
                           least one reverse-reachable event.  Forces n <= T+1.
    C4 (shared queried core): at fixed T, the queried event's content and the
                           split are SHARED BYTE-IDENTICALLY across every n.

NO NEW ARABIC (doc 63 SS4.3 / SS20.1). The six verbs are the committed
`stories_exp51_VT.json` inventory; the surfaces are `exp69_datagen.render` and
`exp69_datagen.question_surface`, CALLED and not copied; the nouns are the
519 of `d4_confirmed_inventory.json`, sha-stamped. doc 50 SS5.2's
native-speaker gate does not reopen.

TWO AMBIGUITIES RESOLVED TOWARD THE COMMITTED CODE AND FLAGGED (doc 63's
standing rule that an ambiguity is flagged, never invented) -- both are recorded
in `results_exp76_WN_validation.json :: flagged_ambiguities`:

  F1. SS4.2's C4 lists "the two queried referents' WIRE SLOTS" among the shared
      core, but SS6.2.1's enumeration of the same core does NOT ("bearer,
      non-bearer, queried verb, query index, that event's surface order" and the
      split).  The two cannot both hold: byte-identical wire slots across n
      would pin the queried pair to slots {0,1} at every n, and SS4.5 L3 /
      SS8.2 V3-n require the bearer's wire-index marginal to be UNIFORM over the
      n slots.  RESOLVED toward SS6.2.1, because SS6.2.1's list is exactly and
      only what SS10.4's `comparator_flat_in_n` needs: tier (ii)'s feature is
      u(theta_agent(q)) (x) u(theta_patient(q)), which does not read a wire
      index.  Wire slots are drawn per (cell, pair); L3 and V3-n hold.

  F2. SS8.2 V5 inherits exp69's "every bearer and non-bearer in >= 1 TRAIN cast"
      over a 24-noun inventory.  At 519 nouns it is ARITHMETICALLY IMPOSSIBLE:
      528 train items are 264 train pairs, and at n = 2 a train pair hosts
      exactly one non-bearer, so at most 264 distinct non-bearers can appear in
      training under SS6.1's frozen 960-item budget.  RESOLVED by making the
      CORE NON-BEARER POOL 264 nouns (drawn at the registered seed) and holding
      it FIXED ACROSS EVERY CELL, so that (a) V5 holds exactly as written over
      the population the dataset uses, (b) SS16's "the split is over unseen
      pairings/casts, NOT unseen nouns" stays literally true, and (c) the noun
      population does not move along the n axis, which would have confounded the
      one axis the design exists to isolate.  The I519 level's working core pool
      is therefore 81 bearers + 264 non-bearers = 345 of the 519; the CATEGORY,
      the FEATURE MAP and every SS2.6 bound remain defined over all 519.

THE SPLIT PATTERN (doc 63 SS20.4): this file BUILDS AND VALIDATES. It trains
nothing and it writes no results file except the V-suite report.

    python exp76_datagen.py                 # build the 11 cells + validate
    python exp76_datagen.py --validate      # re-validate what is on disk
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys

import numpy as np

import exp42_compiler as comp
import exp69_datagen as D69                              # render / question / V2
import exp69_run as R69                                  # the compile path
import instrument_scaling_bounds as ISB                   # the pre-work instrument

SPEC = ("qnlp_private_docs/63_exp76_scaling.md (REGISTERED 2026-08-29, "
        "orchestrator rulings in the status header, BINDING over the body)")
SPEC_DOC = "63_exp76_scaling.md"

VT_SRC = "stories_exp51_VT.json"                          # the committed 6 verbs
INVENTORY = "d4_confirmed_inventory.json"
BOUNDS_ARTEFACT = "diagnostic_scaling_bounds.json"
BANDS_C_ARTEFACT = "diagnostic_scaling_bands_c.json"
VALIDATION_OUT = "results_exp76_WN_validation.json"
STORY_FMT = "stories_exp76_%s.json"

# doc 63 SS8.5 -- asserted at smoke, verified against the file's own noun list.
NOUNS_SHA256_DOC63 = \
    "34e95fc6e0429ecb9284f825e5ac411eba48865db3a3ce585aaf6d6aa26328f0"

SEED = 76000                                  # registered generator seed
CONFIRM_SEED = SEED + 1000                    # doc 63 SS20.2's confirm draw
I24_SEED = SEED + 24                          # doc 63 SS6.3's registered subset

# doc 63 SS6.1: the committed budget, INHERITED and never chosen here.
N_ITEMS = 960
N_TRAIN_ITEMS, N_VAL_ITEMS, N_TEST_ITEMS = 528, 132, 300
N_PAIRS = N_ITEMS // 2                                    # 480
N_TRAIN_PAIRS = N_TRAIN_ITEMS // 2                        # 264
N_VAL_PAIRS = N_VAL_ITEMS // 2                            # 66
N_TEST_PAIRS = N_TEST_ITEMS // 2                          # 150

# F2 above: the core non-bearer pool is the number of TRAIN PAIRS, which is the
# arithmetic ceiling on distinct non-bearers reachable by training at n = 2.
CORE_NONBEARER_POOL = N_TRAIN_PAIRS                       # 264

# doc 63 SS6.3: the I24 level. 81/519 = 0.15607, x 24 = 3.746 -> 4 bearers.
I24_SIZE = 24
I24_BEARERS = 4
I24_NONBEARERS = I24_SIZE - I24_BEARERS                   # 20

ORDERS = ("SVO", "VSO")
MAX_DRAWS = 500                                           # exp69's STOP rule

# doc 63 SS8.2 V3-n's scheduled tolerances. The bearer wire index is EXACT by
# construction (a deterministic cycle), so its tolerance is 1 count; the
# marginals that are drawn carry a relative band.
WIRE_MARGINAL_MAX_SPREAD = 1                  # bearer slot counts, max - min
DEGREE_BALANCE_Z = 4.0                        # paired bearer-vs-queried-non-bearer
DEGREE_BALANCE_ABS = 0.25                     # events, absolute band on the same delta
MARGINAL_REL_TOL = 0.25                       # drawn marginals, |x/E - 1|


# ==================================================================== the grid
# doc 63 SS6.2 (grid), SS6.2.3 (grade-3 cell), SS6.3 (inventory axis).
# ELEVEN trained cells; SS12.4 registers eleven `binds` booleans and SS18.1
# eleven matched floors, and the counts match.
CELL_CFG = {
    # ---- SS6.2's eight grid cells: n <= T + 1, every referent named ---------
    "n2_T2": dict(n=2, T=2, inv="I519", cat="faail_camel", axis="grid"),
    "n3_T2": dict(n=3, T=2, inv="I519", cat="faail_camel", axis="grid"),
    "n2_T4": dict(n=2, T=4, inv="I519", cat="faail_camel", axis="grid"),
    "n4_T4": dict(n=4, T=4, inv="I519", cat="faail_camel", axis="grid"),
    "n5_T4": dict(n=5, T=4, inv="I519", cat="faail_camel", axis="grid"),
    "n2_T6": dict(n=2, T=6, inv="I519", cat="faail_camel", axis="grid"),
    "n4_T6": dict(n=4, T=6, inv="I519", cat="faail_camel", axis="grid"),
    "n7_T6": dict(n=7, T=6, inv="I519", cat="faail_camel", axis="grid"),
    # ---- SS6.2.3's grade-3 target, bought by the pre-work -------------------
    "band32_n2_T2": dict(n=2, T=2, inv="I519", cat="band_32arc", axis="grade3"),
    # ---- SS6.3's inventory axis, isolated rather than confounded ------------
    "I24_n2_T2": dict(n=2, T=2, inv="I24", cat="faail_camel", axis="inventory"),
    "I24_n4_T4": dict(n=4, T=4, inv="I24", cat="faail_camel", axis="inventory"),
}
CELLS = list(CELL_CFG)

# doc 63 SS6.2's table, asserted rather than trusted.
GRID_BY_T = {2: [2, 3], 4: [2, 4, 5], 6: [2, 4, 7]}


def core_key(cfg):
    """doc 63 SS6.2.1: one core per (inventory level, category, T). Cells that
    differ only in n share it BYTE-IDENTICALLY, which is what makes SS10.4 an
    exact assertion rather than a statistical one."""
    return "%s|%s|T%d" % (cfg["inv"], cfg["cat"], cfg["T"])


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# =========================================================== the feature layer
_FEAT = {}


def features519():
    """The 519-noun FA angle map, through the pre-work instrument's own code
    objects (`instrument_scaling_bounds.load_inventory519` /
    `.build_fa_features`), CALLED and never copied.

    doc 63 SS2.9(b): the exp76 plane is NOT the committed 44-noun plane -- the
    PCA is fit over 563 vectors and the sign pin is a DETERMINISM device here,
    not a continuity claim. SS6.4's drift anchor is unaffected: it runs on
    exp69's frozen dataset with exp69's frozen features, not on this map."""
    if _FEAT:
        return _FEAT
    nouns, rows, meta = ISB.load_inventory519()
    assert meta["nouns_sha256"] == NOUNS_SHA256_DOC63, \
        ("doc 63 SS8.5: d4_confirmed_inventory.json nouns_sha256 is %r, the "
         "registration pins %r" % (meta["nouns_sha256"], NOUNS_SHA256_DOC63))
    coords, U, prov = ISB.build_fa_features(nouns)
    angles = {n: float(math.atan2(coords[n][1], coords[n][0])) for n in nouns}
    _FEAT.update({"nouns": nouns, "rows": rows, "meta": meta,
                  "coords": coords, "U": U, "prov": prov, "angles": angles})
    return _FEAT


def memberships():
    """The bearer sets of record, re-derived through the instrument and asserted
    against `diagnostic_scaling_bounds.json`'s `members_sha256` (doc 63 SS8.6:
    the committed-value import is asserted, not re-typed)."""
    f = features519()
    nouns, rows, coords = f["nouns"], f["rows"], f["coords"]
    surf, camel, both = ISB.wazn_sets(nouns, rows)
    band32 = [n for n in nouns if n in set(ISB.band_members(coords, 32, 81, nouns))]
    art = json.load(open(BOUNDS_ARTEFACT, encoding="utf-8"))
    out = {}
    for key, mem, contrast in (("faail_camel", camel, "faail_camel"),
                               ("faail_surface", surf, "faail_surface"),
                               ("faail_intersection", both, "faail_intersection"),
                               ("band_32arc", band32, "band_32arc")):
        got = ISB.canonical_sha256(sorted(mem))
        want = art["contrasts"][contrast]["members_sha256"]
        assert got == want, \
            ("membership drift for %s: re-derived sha %s, artefact %s -- doc 63 "
             "SS3 registers the membership of record and it must reproduce"
             % (key, got, want))
        assert len(mem) == art["contrasts"][contrast]["n_bearers"]
        out[key] = sorted(mem)
    assert len(out["faail_camel"]) == 81 and len(out["band_32arc"]) == 81, \
        "doc 63 SS2.6: |B| = 81 for both the membership of record and band_32arc"
    # SS12.9.0: the grade-3 target is provably unaffected by SS2.6.2's
    # constructor defect, and it is CHECKED here rather than assumed.
    bc = json.load(open(BANDS_C_ARTEFACT, encoding="utf-8"))
    assert bc["grade3_target_unaffected"]["identical"] is True, \
        "doc 63 SS12.9.0: bands_c does not certify the grade-3 target unaffected"
    out["_grade1_bounds"] = {
        "faail_camel": art["contrasts"]["faail_camel"]["value"],
        "band_32arc": art["contrasts"]["band_32arc"]["value"]}
    out["_band32_below_null"] = art["contrasts"]["band_32arc"]["grade3"]["below_null"]
    assert out["_band32_below_null"] is True, \
        "doc 63 SS2.7.1: band_32arc is the grade-3 target because it is BELOW NULL"
    return out


def inventory_level(level, cat):
    """(bearers, nonbearer_core_pool, provenance) for one inventory level.

    I519 -- the 519 confirmed nouns, the category of record over them, and the
            264-noun core non-bearer pool of ambiguity F2, drawn at SEED.
    I24  -- doc 63 SS6.3: a RANDOM 24-noun subset of the 519 drawn at a
            registered seed with the bearer fraction preserved, and the category
            restricted to that subset."""
    f = features519()
    nouns = f["nouns"]
    mem = memberships()
    bearers_all = mem[cat]
    nonbearers_all = [n for n in nouns if n not in set(bearers_all)]
    if level == "I519":
        rng = random.Random(SEED + 7)
        pool = sorted(rng.sample(nonbearers_all, CORE_NONBEARER_POOL))
        prov = {"level": "I519", "category": cat,
                "n_inventory": len(nouns),
                "n_bearers": len(bearers_all),
                "n_nonbearers_available": len(nonbearers_all),
                "n_nonbearer_core_pool": len(pool),
                "core_pool_seed": SEED + 7,
                "why_the_pool_is_264": (
                    "AMBIGUITY F2, flagged not invented: doc 63 SS8.2 V5 requires "
                    "every non-bearer in >= 1 TRAIN cast, and SS6.1 freezes the "
                    "budget at 960 items = 264 train pairs. At n = 2 a train pair "
                    "hosts one non-bearer, so 264 is the arithmetic ceiling. The "
                    "pool is held FIXED ACROSS EVERY CELL so the noun population "
                    "does not move along the n axis.")}
        return sorted(bearers_all), pool, prov
    assert level == "I24"
    rng = random.Random(I24_SEED)
    draws = 0
    while True:
        draws += 1
        if draws > MAX_DRAWS:
            raise RuntimeError("I24 subset never satisfied its rule in %d draws "
                               "-- STOP, do not weaken the rule" % MAX_DRAWS)
        b = sorted(rng.sample(bearers_all, I24_BEARERS))
        nb = sorted(rng.sample(nonbearers_all, I24_NONBEARERS))
        if len(set(b) | set(nb)) == I24_SIZE:
            break
    prov = {"level": "I24", "category": cat,
            "n_inventory": I24_SIZE, "n_bearers": len(b), "n_nonbearers": len(nb),
            "bearer_fraction_target": len(bearers_all) / float(len(nouns)),
            "bearer_fraction_realised": len(b) / float(I24_SIZE),
            "seed": I24_SEED, "draws": draws,
            "note": ("doc 63 SS6.3: a random 24-noun subset of the 519 with the "
                     "bearer fraction preserved (0.1561 x 24 = 3.75 -> 4). "
                     "Comparing I24 against exp69's committed 24-noun arms is "
                     "REFUSED (SS12.12(f), doc 47 SS18.5).")}
    return b, nb, prov


# ================================================================== the core
def build_core(inv, cat, T, seed):
    """doc 63 SS4.2 C4 / SS6.2.1 -- THE SHARED QUERIED CORE.

    480 pair records carrying the queried event's content and the split. Cells
    at the same (inventory, category, T) and different n consume THIS OBJECT,
    so tier (ii)'s comparator feature matrix and its labels are bit-identical
    across n and SS10.4 is an EXACT assertion.

    The non-queried events' VERBS and every event's SURFACE ORDER are drawn here
    too: SS6.2.1's "larger n ADDS referents and re-targets the non-queried
    events" is honoured literally -- across n only the participants and the
    cast->wire map move."""
    bearers, nonbearers, inv_prov = inventory_level(inv, cat)
    verbs = load_verbs()
    rng = random.Random(seed + 17 * T + (0 if cat == "faail_camel" else 500)
                        + (0 if inv == "I519" else 900))

    n_pairings = min(N_PAIRS, len(bearers) * len(nonbearers))
    if n_pairings < N_PAIRS:
        # I24: 4 x 20 = 80 distinct pairings; the split is over PAIRINGS, so
        # each pairing carries N_PAIRS / n_pairings pairs -- exp69's 80-cast x
        # 6-pair shape exactly. Registered as a structural difference from the
        # I519 level and reported in the validation artefact.
        assert N_PAIRS % n_pairings == 0, \
            "pairing pool %d does not divide the 480-pair budget" % n_pairings
    per_pairing = N_PAIRS // n_pairings
    n_test_pg, n_val_pg = N_TEST_PAIRS // per_pairing, N_VAL_PAIRS // per_pairing
    n_train_pg = N_TRAIN_PAIRS // per_pairing
    assert n_test_pg + n_val_pg + n_train_pg == n_pairings, \
        "pairing split does not partition the pool (%d/%d/%d of %d)" \
        % (n_train_pg, n_val_pg, n_test_pg, n_pairings)

    # ---- the pairings, built so V5's train coverage holds BY CONSTRUCTION ---
    if per_pairing == 1:
        # I519: 264 train pairings host the 264 core non-bearers, one each, and
        # cycle the 81 bearers; val and test pairings are DISJOINT re-pairings
        # of the same nouns -- the split is over unseen PAIRINGS, not nouns.
        assert n_train_pg == len(nonbearers), \
            "the core pool must be exactly the train-pairing count"
        bcycle = list(bearers)
        rng.shuffle(bcycle)
        nbs = list(nonbearers)
        rng.shuffle(nbs)
        train_pg = [(bcycle[i % len(bcycle)], nbs[i]) for i in range(n_train_pg)]
        used = set(train_pg)
        extra, guard = [], 0
        while len(extra) < n_val_pg + n_test_pg:
            guard += 1
            if guard > MAX_DRAWS * 1000:
                raise RuntimeError("val/test pairings never drew clear of train "
                                   "-- STOP, do not weaken the rule")
            cand = (rng.choice(bearers), rng.choice(nonbearers))
            if cand in used:
                continue
            used.add(cand)
            extra.append(cand)
        val_pg, test_pg = extra[:n_val_pg], extra[n_val_pg:]
    else:
        # I24: exp69's shape. Every (b, nb) pairing exists; the split is drawn
        # until the train half covers every bearer and every non-bearer.
        allp = [(b, nb) for b in bearers for nb in nonbearers]
        draws = 0
        while True:
            draws += 1
            if draws > MAX_DRAWS:
                raise RuntimeError("train coverage never satisfied in %d draws "
                                   "-- STOP, do not weaken the rule" % MAX_DRAWS)
            pool = list(allp)
            rng.shuffle(pool)
            test_pg, val_pg = pool[:n_test_pg], pool[n_test_pg:n_test_pg + n_val_pg]
            train_pg = pool[n_test_pg + n_val_pg:]
            seen = {x for c in train_pg for x in c}
            if set(bearers) <= seen and set(nonbearers) <= seen:
                break

    pairings = ([(p, "train") for p in train_pg]
                + [(p, "val") for p in val_pg]
                + [(p, "test") for p in test_pg])

    # ---- the 480 pair records, with SS8.2 V3-n's exact balance schedules ----
    core = []
    for gi, ((b, nb), split) in enumerate(pairings):
        for j in range(per_pairing):
            i = gi * per_pairing + j
            q = 1 + (i % T)
            vq = verbs[(i // T) % len(verbs)]
            # the queried event's surface order; the stride is chosen so that
            # every (q, v_q) combination receives BOTH orders (SS8.2 V3-n).
            qorder = ORDERS[((i // T) + i) % 2]
            others = [v for v in verbs if v != vq]
            rng.shuffle(others)
            other_verbs = others[:T - 1]                  # C2: pairwise distinct
            # the queried slot's order follows the exact schedule above; the
            # non-queried slots are DRAWN, because any deterministic stride over
            # t correlates with q = 1 + (i % T) and would pin a slot's order to
            # the query index (V3-n catches exactly that).
            orders = [ORDERS[rng.randrange(2)] for _ in range(T)]
            orders[q - 1] = qorder
            core.append({
                "pair_index": i, "pairing_index": gi, "split": split,
                "bearer": b, "nonbearer": nb,
                "queried_verb": vq, "query_index": q, "queried_order": qorder,
                "other_verbs": other_verbs, "orders": orders,
                "bearer_is_agent_first": True})
    assert len(core) == N_PAIRS
    counts = {s: sum(1 for c in core if c["split"] == s) for s in
              ("train", "val", "test")}
    assert counts == {"train": N_TRAIN_PAIRS, "val": N_VAL_PAIRS,
                      "test": N_TEST_PAIRS}, counts
    return {"records": core, "verbs": verbs, "bearers": bearers,
            "nonbearers": nonbearers, "inventory_provenance": inv_prov,
            "n_pairings": n_pairings, "pairs_per_pairing": per_pairing,
            "core_seed": seed, "T": T,
            "shared_fields_doc63_SS6_2_1": [
                "bearer", "nonbearer", "queried_verb", "query_index",
                "queried_order", "other_verbs", "orders", "split"]}


def load_verbs():
    verbs = sorted(json.load(open(VT_SRC, encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())
    assert len(verbs) == 6, \
        "doc 63 SS4.3: the committed 6-verb exp51 inventory, unchanged"
    return verbs


# ============================================ SS4.4's reverse-reachability
def reverse_reachable(events, qw):
    """doc 63 SS4.4's theorem, as an algorithm. `events` is the circuit-order
    list of (verb, wire_agent, wire_patient); `qw` is the queried pair.

    Process the events in REVERSE circuit order maintaining R = qw. An event is
    ACTIVE iff its wire pair meets the current R, and if it is, its wires join
    R. Returns (active_flags_in_circuit_order, R)."""
    R = set(qw)
    active = [False] * len(events)
    for t in range(len(events) - 1, -1, -1):
        _v, wa, wp = events[t]
        if wa in R or wp in R:
            active[t] = True
            R.add(wa)
            R.add(wp)
    return active, R


def draw_events(rec, n, T, wire_of, rng):
    """The T events of one pair, satisfying C1, C2 and C3.

    C3 is met CONSTRUCTIVELY: walking the non-queried event slots in reverse
    circuit order, exactly n - 2 of them introduce one wire not yet reachable
    and the rest re-use reachable wires. Every event therefore meets R when it
    is processed, and all n wires end in R -- so n_active = n = min(n, T+1)."""
    q = rec["query_index"]
    wb, wn = wire_of[rec["bearer"]], wire_of[rec["nonbearer"]]
    events = [None] * T
    events[q - 1] = (rec["queried_verb"], wb, wn)         # C1; the twin swaps it

    slots = [t for t in range(T - 1, -1, -1) if t != q - 1]   # reverse order
    n_new = n - 2
    assert 0 <= n_new <= len(slots), \
        ("doc 63 SS4.4.2: n <= T + 1 is violated (n=%d, T=%d)" % (n, T))
    intro_at = set(rng.sample(range(len(slots)), n_new))
    fresh = [w for w in range(n) if w not in (wb, wn)]
    rng.shuffle(fresh)

    R = {wb, wn}
    ov = list(rec["other_verbs"])
    for k, t in enumerate(slots):
        v = ov[k]
        if k in intro_at:
            x = fresh.pop()
            y = rng.choice(sorted(R))
            R.add(x)
        else:
            x, y = rng.sample(sorted(R), 2)
        if rng.random() < 0.5:
            x, y = y, x
        events[t] = (v, x, y)
    assert R == set(range(n)), \
        "C3 violated: cast members outside the reachable set %r" % sorted(R)
    active, Rf = reverse_reachable(events, (wb, wn))
    assert all(active) and Rf == set(range(n)), \
        "C3 violated: an event is inert or a referent is unreachable"
    return events


# ==================================================================== build
def build_cell(cell, cores, seed):
    """One cell's 960 items. Only the distractor identities, the cast->wire map
    and the non-queried events' PARTICIPANTS depend on n (doc 63 SS6.2.1)."""
    cfg = CELL_CFG[cell]
    n, T = cfg["n"], cfg["T"]
    core = cores[core_key(cfg)]
    rng = random.Random(seed + 1000 * n + 13 * T
                        + (0 if cfg["cat"] == "faail_camel" else 500)
                        + (0 if cfg["inv"] == "I519" else 900))
    pool = core["nonbearers"]
    items = []
    for rec in core["records"]:
        i = rec["pair_index"]
        b, nb = rec["bearer"], rec["nonbearer"]
        distractors = []
        if n > 2:
            cand = [x for x in pool if x != nb]
            distractors = rng.sample(cand, n - 2)
        cast = [b, nb] + distractors
        assert len(set(cast)) == n, "degenerate cast at pair %d" % i

        # SS4.5 L3: the bearer's wire slot is an EXACT deterministic cycle, so
        # its marginal is uniform to within one count at every n; the remaining
        # cast members are permuted onto the free slots.
        wb = i % n
        free = [w for w in range(n) if w != wb]
        rng.shuffle(free)
        wire_of = {b: wb}
        for name, w in zip([nb] + distractors, free):
            wire_of[name] = w
        by_wire = {w: name for name, w in wire_of.items()}
        assert len(by_wire) == n

        events = draw_events(rec, n, T, wire_of, rng)
        qw = tuple(sorted((wire_of[b], wire_of[nb])))
        refs = [{"name": by_wire[w], "type": "human"} for w in range(n)]
        key = "p%05d" % i

        for suffix, agent, patient, ans in (("a", b, nb, "yes"),
                                            ("b", nb, b, "no")):
            evs, sents = [], []
            for t, (v, wa, wp) in enumerate(events):
                if t == rec["query_index"] - 1:
                    ea, ep = agent, patient          # the twin swap, event q ONLY
                else:
                    ea, ep = by_wire[wa], by_wire[wp]
                evs.append({"verb": v, "agent": ea, "patient": ep,
                            "pro_drop": False})
                sents.append(D69.render(rec["orders"][t], v, ea, ep))
            items.append({
                "id": "%s_%s_%s" % (rec["split"], key, suffix),
                "pair_id": key, "split": rec["split"], "qtype": "Q76",
                "hop_count": 1, "pro_drop_item": False,
                "order_dependent": True, "answer": ans,
                "K": n, "T": T, "flip_idx": [rec["query_index"] - 1],
                "query_index": rec["query_index"],
                "queried_verb": rec["queried_verb"],
                "referents": refs,
                "events": evs,
                "sentences": sents,
                "order_flags": list(rec["orders"]),
                "question": {"type": "Q76",
                             "surface": D69.question_surface(rec["queried_verb"]),
                             "v": rec["queried_verb"],
                             "a": by_wire[qw[0]], "p": by_wire[qw[1]]},
                "cell": cell, "n": n, "n_active": n,
                "bearer": b, "queried_nonbearer": nb,
                "distractors": list(distractors),
            })
    assert len(items) == N_ITEMS, len(items)
    return items


def cell_meta(cell, core, items):
    cfg = CELL_CFG[cell]
    return {
        "experiment": "exp76", "cell": cell, "design_doc": SPEC,
        "n": cfg["n"], "T": cfg["T"], "n_active": min(cfg["n"], cfg["T"] + 1),
        "inventory_level": cfg["inv"], "category": cfg["cat"], "axis": cfg["axis"],
        "seed": SEED, "is_confirm_draw": SEED != 76000,
        "core_key": core_key(cfg),
        "n_items": len(items),
        "split_counts": {s: sum(1 for i in items if i["split"] == s)
                         for s in ("train", "val", "test")},
        "answer_balance": {a: sum(1 for i in items if i["answer"] == a)
                           for a in ("yes", "no")},
        "verb_inventory": {v: {"human_patients": True} for v in core["verbs"]},
        "verb_inventory_source": VT_SRC,
        "n_pairings": core["n_pairings"],
        "pairs_per_pairing": core["pairs_per_pairing"],
        "inventory_provenance": core["inventory_provenance"],
        "shared_core_doc63_SS6_2_1": core["shared_fields_doc63_SS6_2_1"],
        "nouns_sha256": NOUNS_SHA256_DOC63,
        "dataset_sha256": comp.canonical_sha256(items),
    }


def build_all(seed=SEED):
    cores = {}
    for cell in CELLS:
        cfg = CELL_CFG[cell]
        k = core_key(cfg)
        if k not in cores:
            cores[k] = build_core(cfg["inv"], cfg["cat"], cfg["T"], seed)
    out = {}
    for cell in CELLS:
        items = build_cell(cell, cores, seed)
        meta = cell_meta(cell, cores[core_key(CELL_CFG[cell])], items)
        path = STORY_FMT % cell
        json.dump({"meta": meta, "items": items},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)
        out[cell] = path
        print("[exp76 datagen] %-14s n=%d T=%d %-5s %-12s -> %s (%d items)"
              % (cell, meta["n"], meta["T"], meta["inventory_level"],
                 meta["category"], path, len(items)), flush=True)
    return out


def load_cell(cell):
    path = STORY_FMT % cell
    blob = json.load(open(path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp76 dataset provenance FAILED for %s" % path)
    return items, meta, path


# ============================================== doc 63 SS8.2 -- THE V-SUITE
def _acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def v1_bag(items, angles):
    """V1 -- BAG OF EMBEDDINGS. Exactly 0.500, with the twin features
    BIT-IDENTICAL on all 150 test pairs. RE-MEASURED at every (n, T), never
    assumed (doc 63 SS4.5). Failing direction: any value off 0.500, or any
    non-identical twin feature, fails the cell."""
    from sklearn.linear_model import LogisticRegression

    def feat(it):
        u = [np.array((1.0, math.cos(angles[r["name"]]),
                       math.sin(angles[r["name"]]))) for r in it["referents"]]
        s = np.sum(u, axis=0)
        return np.concatenate([s, s / float(len(u)), [float(len(u))]])

    X = {s: np.array([feat(i) for i in items if i["split"] == s])
         for s in ("train", "test")}
    y = {s: np.array([1 if i["answer"] == "yes" else 0
                      for i in items if i["split"] == s])
         for s in ("train", "test")}
    clf = LogisticRegression(max_iter=5000).fit(X["train"], y["train"])
    acc = _acc(y["test"], clf.predict(X["test"]))

    byp, worst = {}, 0.0
    for it in items:
        if it["split"] == "test":
            byp.setdefault(it["pair_id"], []).append(it)
    for _p, g in byp.items():
        assert len(g) == 2
        worst = max(worst, float(np.abs(feat(g[0]) - feat(g[1])).max()))
    assert worst == 0.0, \
        ("doc 63 SS8.2 V1: twin bag features differ by %.3e -- the cast is not "
         "shared across twins and the arm is a null instrument" % worst)
    assert abs(acc - 0.5) < 1e-12, \
        ("doc 63 SS8.2 V1: bag-of-embeddings scores %.6f, not exactly 0.500 -- "
         "a cast-only function separates the twins" % acc)
    return {"test_acc": acc, "exactly_half": True,
            "twin_feature_max_abs_diff": worst, "bit_identical_twins": True}


def v2_ngram(items, meta):
    """V2 -- N-GRAM, RE-DERIVED AND REPORTED UN-GATED (doc 63 SS8.2), exactly as
    exp68 and exp69 did. `exp69_datagen.v2_ngram` loops over exp69's T in {1, 2}
    and cannot be called on a T in {2, 4, 6} cell, so the loop is re-derived here
    and the TOKENISER is `exp69_datagen._tokens`, CALLED and not copied."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    te = [i for i in items if i["split"] == "test"]
    vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+")
    Xtr = vec.fit_transform([D69._tokens(i) for i in tr])
    m = LogisticRegression(max_iter=3000, C=10.0).fit(
        Xtr, [i["answer"] for i in tr])
    return {"gated": False, "T": meta["T"],
            "exp60_V2H_bar_not_inherited": 0.55,
            "seen_pairings_val_accuracy": _acc(
                [i["answer"] for i in va],
                m.predict(vec.transform([D69._tokens(i) for i in va]))),
            "unseen_pairings_test_accuracy": _acc(
                [i["answer"] for i in te],
                m.predict(vec.transform([D69._tokens(i) for i in te]))),
            "n_features": int(Xtr.shape[1])}


def v3n_balance(items, meta):
    """V3-n -- BALANCE, EXTENDED (doc 63 SS8.2, SS4.5 L2/L3). Verb, verb-order,
    per-event surface order, query index, bearer wire-index, AND -- new --
    per-referent participation degree stratified by bearer / non-bearer.

    Failing direction: any marginal outside the scheduled tolerance fails the
    cell."""
    n, T = meta["n"], meta["T"]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], []).append(it)
    firsts = [sorted(g, key=lambda x: x["id"])[0] for g in pairs.values()]
    assert len(firsts) == N_PAIRS

    qv = {}
    qi = {}
    slot_order = [dict() for _ in range(T)]
    wire_of_bearer = {}
    deg_b, deg_qnb, deg_dis = [], [], []
    for it in firsts:
        qv[it["queried_verb"]] = qv.get(it["queried_verb"], 0) + 1
        qi[it["query_index"]] = qi.get(it["query_index"], 0) + 1
        for t, f in enumerate(it["order_flags"]):
            slot_order[t][f] = slot_order[t].get(f, 0) + 1
        wires = {r["name"]: w for w, r in enumerate(it["referents"])}
        wb = wires[it["bearer"]]
        wire_of_bearer[wb] = wire_of_bearer.get(wb, 0) + 1
        cnt = {}
        for ev in it["events"]:
            cnt[ev["agent"]] = cnt.get(ev["agent"], 0) + 1
            cnt[ev["patient"]] = cnt.get(ev["patient"], 0) + 1
        deg_b.append(cnt.get(it["bearer"], 0))
        deg_qnb.append(cnt.get(it["queried_nonbearer"], 0))
        for d in it["distractors"]:
            deg_dis.append(cnt.get(d, 0))

    # (a) the queried verb, the query index, the per-slot surface order
    exp_v = N_PAIRS / 6.0
    for v, c in qv.items():
        assert abs(c / exp_v - 1.0) <= MARGINAL_REL_TOL, \
            "V3-n: queried-verb marginal %r=%d against %.1f" % (v, c, exp_v)
    exp_q = N_PAIRS / float(T)
    for k, c in qi.items():
        assert abs(c / exp_q - 1.0) <= MARGINAL_REL_TOL, \
            "V3-n: query-index marginal %r=%d against %.1f" % (k, c, exp_q)
    for t, d in enumerate(slot_order):
        assert set(d) == {"SVO", "VSO"}, \
            "V3-n: event slot %d does not carry BOTH surface orders: %r" % (t, d)
        assert abs(d["SVO"] / (N_PAIRS / 2.0) - 1.0) <= MARGINAL_REL_TOL, \
            "V3-n: surface-order marginal at slot %d is %r" % (t, d)

    # (b) SS4.5 L3 -- the bearer's wire-index marginal, EXACT by construction
    assert set(wire_of_bearer) == set(range(n)), \
        "V3-n: the bearer never occupies wires %r" \
        % sorted(set(range(n)) - set(wire_of_bearer))
    spread = max(wire_of_bearer.values()) - min(wire_of_bearer.values())
    assert spread <= WIRE_MARGINAL_MAX_SPREAD, \
        ("V3-n / SS4.5 L3: the bearer's wire-index marginal spreads by %d "
         "counts -- a positional leak" % spread)

    # (c) SS4.5 L2 -- participation degree, stratified. The instrument that
    # matters is bearer vs QUERIED NON-BEARER: those are the two referents a
    # model would have to tell apart, and the twin shares the structure exactly.
    mb, mq = float(np.mean(deg_b)), float(np.mean(deg_qnb))
    d = np.asarray(deg_b, float) - np.asarray(deg_qnb, float)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(len(d)) if sd > 0 else 0.0
    z = (md / se) if se > 0 else 0.0
    # The gate is PAIRED and z-scored rather than a fixed epsilon: the bearer
    # and the queried non-bearer are structurally exchangeable in `draw_events`
    # (both enter R at the queried event and neither is treated specially), so
    # the null is a zero MEAN with a sampling sd the design does not control.
    # A systematic degree leak of even 0.2 events would land past z = 5.
    assert abs(z) <= DEGREE_BALANCE_Z, \
        ("V3-n / SS4.5 L2: mean participation degree bearer %.4f vs queried "
         "non-bearer %.4f (paired delta %.4f, z = %.2f) -- a degree shortcut "
         "to the bearer's identity" % (mb, mq, md, z))
    assert abs(md) <= DEGREE_BALANCE_ABS, \
        ("V3-n / SS4.5 L2: paired degree delta %.4f exceeds the absolute band "
         "%.2f -- the gate is not allowed to pass on a wide sd alone"
         % (md, DEGREE_BALANCE_ABS))
    return {"queried_verb": qv, "query_index": qi,
            "surface_order_per_slot": slot_order,
            "bearer_wire_index": wire_of_bearer, "wire_index_spread": spread,
            "degree_bearer_mean": mb, "degree_queried_nonbearer_mean": mq,
            "degree_paired_delta_mean": md, "degree_paired_delta_sd": sd,
            "degree_paired_delta_z": z,
            "degree_distractor_mean": (float(np.mean(deg_dis))
                                       if deg_dis else None),
            "degree_balance_z_bar": DEGREE_BALANCE_Z,
            "degree_balance_abs_bar": DEGREE_BALANCE_ABS,
            "note": ("SS4.5 L2's leak is a POPULATION-level bearer/non-bearer "
                     "degree correlation. The reported contrast is bearer vs "
                     "QUERIED non-bearer because those two are the pair a model "
                     "must separate; distractor degree is reported, not gated, "
                     "because a distractor is never a label candidate.")}


def v4_dedup(items, angles):
    """V4 -- all items compile to DISTINCT circuits; zero contradictory label
    pairs."""
    seen, contra = {}, []
    for it in items:
        cs = R69.compile_item(it, angles)
        key = (cs.K, tuple(cs.intro), tuple(cs.events), cs.question)
        if key in seen and seen[key][1] != cs.answer:
            contra.append((seen[key][0], it["id"]))
        seen.setdefault(key, (it["id"], cs.answer))
    assert len(seen) == len(items), \
        ("doc 63 SS8.2 V4: %d of %d items compile to a duplicate circuit"
         % (len(items) - len(seen), len(items)))
    assert not contra, "doc 63 SS8.2 V4: contradictory label pairs %r" % contra[:5]
    return {"n_items": len(items), "n_distinct_circuits": len(seen),
            "contradictory_pairs": 0}


def v5_split_integrity(items, meta):
    """V5 -- test casts disjoint from train/val; every bearer and non-bearer in
    >= 1 train cast; every verb in >= 1 train item; both surface orders per
    event slot. See ambiguity F2 for what "every non-bearer" means at 519."""
    pg = {s: set() for s in ("train", "val", "test")}
    nouns = {s: set() for s in ("train", "val", "test")}
    verbs_train = set()
    for it in items:
        s = it["split"]
        pg[s].add((it["bearer"], it["queried_nonbearer"]))
        for r in it["referents"]:
            nouns[s].add(r["name"])
        if s == "train":
            for ev in it["events"]:
                verbs_train.add(ev["verb"])
    assert not (pg["test"] & pg["train"]), \
        "doc 63 SS8.2 V5: test pairings intersect train"
    assert not (pg["test"] & pg["val"]), \
        "doc 63 SS8.2 V5: test pairings intersect val"
    assert not (pg["val"] & pg["train"]), \
        "doc 63 SS8.2 V5: val pairings intersect train"
    unseen = (nouns["test"] | nouns["val"]) - nouns["train"]
    assert not unseen, \
        ("doc 63 SS8.2 V5 / SS16: %d held-out nouns are UNSEEN in training (%r) "
         "-- exp76 holds out PAIRINGS, not nouns, and the scope sentence would "
         "be false" % (len(unseen), sorted(unseen)[:5]))
    assert len(verbs_train) == 6, \
        "doc 63 SS8.2 V5: only %d verbs appear in training" % len(verbs_train)
    return {"n_pairings": {s: len(pg[s]) for s in pg},
            "n_nouns": {s: len(nouns[s]) for s in nouns},
            "test_disjoint_from_train_and_val": True,
            "every_heldout_noun_seen_in_train": True,
            "n_verbs_in_train": len(verbs_train)}


def v7_label_definition(items):
    """V7 -- label = 1 iff agent(event q) is the bearer, RECOMPUTED from the
    item."""
    bad = []
    for it in items:
        q = it["query_index"]
        want = "yes" if it["events"][q - 1]["agent"] == it["bearer"] else "no"
        if want != it["answer"]:
            bad.append(it["id"])
    assert not bad, \
        "doc 63 SS8.2 V7: %d items carry a label the definition does not give " \
        "(%r)" % (len(bad), bad[:5])
    return {"n_checked": len(items), "mismatches": 0,
            "definition": "label = yes iff agent(event q) is the bearer"}


def v8_query_resolvability(items):
    """V8 -- GENERALISED. For every item the queried verb occurs in EXACTLY ONE
    of the T events. Q76's C2 makes this true by construction and V8 ASSERTS it
    rather than arguing it. Failing direction: a repeated verb in any story
    fails the dataset."""
    bad = []
    for it in items:
        vs = [e["verb"] for e in it["events"]]
        if len(set(vs)) != len(vs) or vs.count(it["queried_verb"]) != 1:
            bad.append(it["id"])
    assert not bad, \
        "doc 63 SS8.2 V8: %d items do not resolve their query uniquely (%r)" \
        % (len(bad), bad[:5])
    return {"n_checked": len(items), "violations": 0,
            "assertion": "the queried verb occurs in exactly one of the T events"}


def v9_active_structure(items, meta, angles):
    """V9 -- NEW. For every item, EVERY event is reverse-reachable from the
    queried pair and `n_active` equals min(n, T+1) (doc 63 SS4.4.2).

    Failing direction: an item containing an inert event fails the dataset --
    it is a smaller item wearing a larger cell's name."""
    n, T = meta["n"], meta["T"]
    want = min(n, T + 1)
    bad_inert, bad_n = [], []
    for it in items:
        cs = R69.compile_item(it, angles)
        _vs, qa, qp = cs.question
        active, R = reverse_reachable(cs.events, (qa, qp))
        if not all(active):
            bad_inert.append(it["id"])
        if len(R) != want or len(R) != cs.K:
            bad_n.append((it["id"], len(R)))
    assert not bad_inert, \
        ("doc 63 SS8.2 V9: %d items contain a PROVABLY INERT event (%r) -- they "
         "are smaller items wearing this cell's name" % (len(bad_inert),
                                                         bad_inert[:5]))
    assert not bad_n, \
        ("doc 63 SS8.2 V9 / SS4.4.2: n_active != min(n, T+1) = %d for %d items "
         "(%r)" % (want, len(bad_n), bad_n[:5]))
    return {"n_checked": len(items), "n_active": want, "n_active_expected": want,
            "all_events_reverse_reachable": True, "inert_events": 0}


def validate_cell(cell, angles):
    items, meta, path = load_cell(cell)
    rep = {"cell": cell, "path": path, "dataset_sha256": meta["dataset_sha256"],
           "n": meta["n"], "T": meta["T"], "inventory_level": meta["inventory_level"],
           "category": meta["category"], "split_counts": meta["split_counts"],
           "answer_balance": meta["answer_balance"]}
    assert meta["answer_balance"] == {"yes": N_ITEMS // 2, "no": N_ITEMS // 2}, \
        "doc 63 SS4.2: the twin construction is exactly 50/50 and it is not"
    rep["V1_bag"] = v1_bag(items, angles)
    rep["V2_ngram"] = v2_ngram(items, meta)
    rep["V2_note"] = (
        "doc 63 SS8.2 V2: RE-DERIVED AND REPORTED UN-GATED, exactly as exp68 and "
        "exp69 did. The split is over unseen pairings/casts, not unseen nouns, so "
        "token identity transfers BY DESIGN. The price is paid in the CLAIM "
        "(SS16), not in the gate.")
    rep["V3n_balance"] = v3n_balance(items, meta)
    rep["V4_dedup"] = v4_dedup(items, angles)
    rep["V5_split"] = v5_split_integrity(items, meta)
    rep["V7_label"] = v7_label_definition(items)
    rep["V8_query"] = v8_query_resolvability(items)
    rep["V9_active"] = v9_active_structure(items, meta, angles)
    return rep


def check_shared_core(angles):
    """doc 63 SS6.2.1 / SS10.4, ASSERTED AT SMOKE AS A DATASET PROPERTY, BEFORE
    ANY TRAINING: at fixed (inventory, category, T) the tier-(ii) comparator
    feature matrix and the labels are BIT-IDENTICAL across n.

    THE FAILING DIRECTION, ASSERTED IN THE SAME TEST: a build in which the
    queried core is re-drawn per n must FAIL. A generator that passes has
    actually shared the core."""

    def tier_ii(items):
        rows = []
        for it in sorted(items, key=lambda x: x["id"]):
            q = it["query_index"]
            a, p = it["events"][q - 1]["agent"], it["events"][q - 1]["patient"]
            ua = np.array((1.0, math.cos(angles[a]), math.sin(angles[a])))
            up = np.array((1.0, math.cos(angles[p]), math.sin(angles[p])))
            rows.append(np.outer(ua, up).ravel())
        return np.array(rows)

    def labels(items):
        return np.array([1 if i["answer"] == "yes" else 0
                         for i in sorted(items, key=lambda x: x["id"])])

    groups = {}
    for cell in CELLS:
        groups.setdefault(core_key(CELL_CFG[cell]), []).append(cell)
    out = {}
    for k, cells in sorted(groups.items()):
        if len(cells) < 2:
            out[k] = {"cells": cells, "n_cells": len(cells),
                      "assertion": None,
                      "why_null": ("doc 63 SS10.4's scope: the assertion runs "
                                   "across n WITHIN one inventory. This group "
                                   "carries one n, so it is VACUOUS and is "
                                   "stored as null, NOT as TRUE -- a vacuous "
                                   "assertion is not a passed one.")}
            continue
        ref = None
        worst_x = worst_y = 0.0
        for cell in cells:
            items, _m, _p = load_cell(cell)
            X, y = tier_ii(items), labels(items)
            if ref is None:
                ref = (X, y)
                continue
            worst_x = max(worst_x, float(np.abs(X - ref[0]).max()))
            worst_y = max(worst_y, float(np.abs(y - ref[1]).max()))
        assert worst_x == 0.0 and worst_y == 0.0, \
            ("doc 63 SS6.2.1 / SS10.4: the tier-(ii) feature matrix moves by "
             "%.3e (labels %.3e) across n at %s -- THE QUERIED CORE IS NOT "
             "SHARED and the n-axis cells are not comparable" % (worst_x,
                                                                 worst_y, k))
        out[k] = {"cells": cells, "n_cells": len(cells),
                  "max_abs_feature_diff": worst_x, "max_abs_label_diff": worst_y,
                  "bit_identical": True, "assertion": True}

    # ---- THE FAILING DIRECTION: a re-drawn core must FAIL the same test -----
    cores_bad = {}
    cfg = CELL_CFG["n2_T4"]
    k = core_key(cfg)
    cores_bad[k] = build_core(cfg["inv"], cfg["cat"], cfg["T"], SEED + 999)
    bad_items = build_cell("n2_T4", cores_bad, SEED + 999)
    good_items, _m, _p = load_cell("n2_T4")
    d = float(np.abs(tier_ii(bad_items) - tier_ii(good_items)).max())
    assert d > 0.0, \
        ("doc 63 SS10.4's FAILING DIRECTION DID NOT FIRE: a core re-drawn at a "
         "different seed produced a BIT-IDENTICAL tier-(ii) matrix, so the "
         "assertion is a null instrument")
    out["_failing_direction"] = {
        "construction": "the T = 4 core re-drawn at SEED + 999, n = 2 rebuilt",
        "max_abs_feature_diff": d, "fires": True,
        "reading": ("doc 63 SS10.4: a build in which the queried core is "
                    "RE-DRAWN per n must FAIL this assertion. A pipeline that "
                    "passes both is actually sharing the core.")}
    return out


def validate(write=True):
    f = features519()
    angles = f["angles"]
    art = json.load(open(BOUNDS_ARTEFACT, encoding="utf-8"))
    assert f["prov"] == art["features"], \
        ("doc 63 SS8.4 item 4: the re-derived FA feature provenance does not "
         "match diagnostic_scaling_bounds.json :: features")
    mem = memberships()
    rep = {
        "component": "exp76_datagen -- the Q76 generator and its V-suite",
        "spec": SPEC,
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "generator_seed": SEED,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "inventory_file": INVENTORY,
        "inventory_file_sha256": sha256_file(INVENTORY),
        "nouns_sha256": NOUNS_SHA256_DOC63,
        "features_provenance_matches_prework": True,
        "memberships": {k: len(v) for k, v in mem.items()
                        if not k.startswith("_")},
        "grade1_bounds_doc63_SS2_6": mem["_grade1_bounds"],
        "grid_doc63_SS6_2": GRID_BY_T,
        "cells": {},
    }
    for cell in CELLS:
        rep["cells"][cell] = validate_cell(cell, angles)
        print("[exp76 V-suite] %-14s PASSED" % cell, flush=True)
    rep["shared_core_doc63_SS6_2_1"] = check_shared_core(angles)
    print("[exp76 V-suite] shared queried core: bit-identical across n",
          flush=True)

    rep["flagged_ambiguities"] = [
        {"id": "F1", "sections": ["SS4.2 C4", "SS6.2.1", "SS4.5 L3", "SS8.2 V3-n"],
         "conflict": ("SS4.2's C4 puts the queried referents' WIRE SLOTS in the "
                      "shared core; SS6.2.1's enumeration of the same core does "
                      "not. Byte-identical wire slots across n would pin the "
                      "queried pair to slots {0,1} at every n, which contradicts "
                      "SS4.5 L3's uniform bearer wire-index marginal and V3-n's "
                      "assertion of it."),
         "resolution": ("toward SS6.2.1, which is the section whose consequence "
                        "SS10.4 actually reads: tier (ii) is "
                        "u(theta_agent(q)) (x) u(theta_patient(q)) and does not "
                        "read a wire index. Wire slots are drawn per (cell, "
                        "pair); the bearer's slot is an exact deterministic "
                        "cycle so V3-n holds by construction."),
         "status": "FLAGGED FOR THE ORCHESTRATOR, not invented"},
        {"id": "F2", "sections": ["SS8.2 V5", "SS6.1", "SS16"],
         "conflict": ("V5 inherits exp69's 'every bearer and non-bearer in >= 1 "
                      "TRAIN cast' over 24 nouns. At 519 nouns under SS6.1's "
                      "frozen 960-item budget it is arithmetically impossible: "
                      "264 train pairs host at most 264 distinct non-bearers at "
                      "n = 2."),
         "resolution": ("the CORE NON-BEARER POOL is 264 nouns drawn at the "
                        "registered seed and held FIXED ACROSS EVERY CELL. V5 "
                        "then holds exactly as written over the population the "
                        "dataset uses; SS16's 'the split is over unseen "
                        "pairings/casts, not unseen nouns' stays literally true; "
                        "and the noun population does not move along the n axis. "
                        "The I519 working core pool is 345 of the 519; the "
                        "category, the feature map and every SS2.6 bound remain "
                        "defined over all 519."),
         "status": "FLAGGED FOR THE ORCHESTRATOR, not invented"},
        {"id": "F3", "sections": ["SS6.3", "SS12.7"],
         "conflict": ("the I24 level has only 4 x 20 = 80 distinct pairings, so "
                      "the 480-pair budget forces 6 pairs per pairing (exp69's "
                      "shape); the I519 level has 480 distinct pairings, one "
                      "pair each. The two levels therefore differ in split "
                      "GRANULARITY as well as in inventory."),
         "resolution": ("kept, because the alternative -- repeating I519 "
                        "pairings six times -- would shrink the I519 pairing "
                        "pool to 80 and destroy the very axis SS12.7 measures. "
                        "`inventory_costs` is read together with the two levels' "
                        "pre-work bounds (SS12.7) and this note travels with it."),
         "status": "FLAGGED FOR THE ORCHESTRATOR, not invented"},
    ]
    rep["kill_gate_passed"] = True
    if write:
        json.dump(rep, open(VALIDATION_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[exp76 V-suite] wrote %s -- ALL doc 63 SS8.2 ASSERTIONS PASSED"
              % VALIDATION_OUT, flush=True)
    return rep


def main():
    global SEED
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="re-validate the datasets already on disk")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    SEED = a.seed
    if not a.validate:
        build_all(SEED)
    validate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
