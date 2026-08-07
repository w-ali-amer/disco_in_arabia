# -*- coding: utf-8 -*-
"""exp42_controls.py -- pre-registered controls C1-C4 (doc 22 SS6).

C1 trained-dial-swap: permute the trained verb->block assignment; pass =
   original assignment STRICTLY above the 95th percentile of permuted
   accuracies -> content_earned. 15 verbs (15! >> enumerable), so 1,000
   sampled permutations per the doc's ">6 verbs" rule. The permutation is
   applied consistently to story events AND the question verb (it is a
   permutation of the ASSIGNMENT). Aggregation across the 5 seeds (doc
   does not fix it; decided pre-run): for every sampled permutation the
   SAME permutation is applied to all seeds and accuracies are averaged;
   the original seed-mean must strictly exceed the 95th percentile of the
   permuted seed-means. No seed selection.

C2 informative null (SCAFFOLD): blocks trained on a disjoint verb
   inventory, evaluated on the held-out verbs' stories via a fixed
   bijection -> must be <= chance + 5pp. NOTE (flagged): exp41 stories mix
   verbs freely, so verb-disjoint TRAINING stories are rare in the frozen
   dataset; C2 needs a dedicated generator run with a restricted verb
   inventory (allowed pre-training under SS2.3; the frozen exp42 dataset
   itself is untouched). c2_scaffold_report() quantifies this.

C3 order-invariance: SVO<->VSO re-rendering changes the compiled circuit
   NOT AT ALL (machine-precision statevector identity for quantum and B1
   -- structured models never see the surface) and changes B2's input
   (behavioral delta recorded).

C4 sentence-scramble: shuffling event order must destroy accuracy on
   order-dependent items (generator-labeled).
"""

import random
from dataclasses import replace

import torch

import exp42_compiler as comp


def _accuracy(model, compiled, verb_map=None):
    correct = 0
    for cs in compiled:
        if model.predict(cs, verb_map) == cs.answer:
            correct += 1
    return correct / len(compiled)


# ----------------------------------------------------------------- C1
def c1_permutation_test(models, compiled_union, verbs, n_perm=1000,
                        seed=4242):
    """`models`: list of trained structured models (5 seeds, same arm).
    Returns the pre-registered verdict dict with content_earned."""
    rng = random.Random(seed)
    orig = sum(_accuracy(m, compiled_union) for m in models) / len(models)
    perm_accs = []
    for i in range(n_perm):
        p = verbs[:]
        while True:
            rng.shuffle(p)
            if p != verbs:                                # exclude identity
                break
        vmap = dict(zip(verbs, p))
        acc = sum(_accuracy(m, compiled_union, vmap)
                  for m in models) / len(models)
        perm_accs.append(acc)
    perm_sorted = sorted(perm_accs)
    q95 = perm_sorted[int(0.95 * n_perm) - 1]
    return {
        "control": "C1 trained-dial-swap",
        "n_permutations": n_perm,
        "original_seed_mean_acc": orig,
        "perm_95th_percentile": q95,
        "perm_max": perm_sorted[-1],
        "perm_mean": sum(perm_accs) / n_perm,
        "content_earned": bool(orig > q95),               # STRICTLY above
    }


# ----------------------------------------------------------------- C2
def c2_verb_split(verb_inventory):
    """Deterministic swap-balanced disjoint split, fixed BEFORE any C2
    training. FLAGGED CHANGE from the earlier alphabetical scaffold split
    (first 8 sorted): that split left verbset B with a single
    swap-capable verb, which is generator-infeasible (Q1 bridge questions
    draw two verbs from the swap pool). Doc 22 SS6 pre-registers only
    disjointness and the <= chance + 5pp criterion; the split rule is
    implementation scaffolding. Rule: sort swap verbs and non-swap verbs
    separately, alternate A/B."""
    swap = sorted(v for v, c in verb_inventory.items() if c["swap"])
    nons = sorted(v for v, c in verb_inventory.items() if not c["swap"])
    va = sorted(swap[0::2] + nons[0::2])
    vb = sorted(swap[1::2] + nons[1::2])
    return va, vb


def c2_bijection(verb_inventory):
    """Fixed kind-matched bijection: B swap verbs -> A swap blocks, B
    non-swap verbs -> A non-swap blocks (so borrowed blocks are at least
    type-applicable; any residual skill is what C2 measures)."""
    va, vb = c2_verb_split(verb_inventory)
    swap_a = [v for v in va if verb_inventory[v]["swap"]]
    non_a = [v for v in va if not verb_inventory[v]["swap"]]
    vmap = {}
    i = j = 0
    for b in vb:
        if verb_inventory[b]["swap"]:
            vmap[b] = swap_a[i % len(swap_a)]
            i += 1
        else:
            vmap[b] = non_a[j % len(non_a)]
            j += 1
    return vmap


def c2_scaffold_report(items, verb_inventory):
    """How feasible is C2 on the frozen dataset? (Answer: it is not --
    this report is the evidence for the dedicated generator runs.)"""
    va, vb = c2_verb_split(verb_inventory)
    va_set = set(va)

    def qverbs(q):                                        # Q0/Q2 vs Q1
        return (q["v"],) if "v" in q else (q["v1"], q["v2"])

    n_a_only = sum(1 for it in items if it["split"] == "train"
                   and all(e["verb"] in va_set for e in it["events"])
                   and all(v in va_set for v in qverbs(it["question"])))
    return {
        "control": "C2 informative-null scaffold",
        "verbset_A": va, "verbset_B": vb,
        "n_train_stories_verbset_A_only": n_a_only,
        "note": "C2 uses dedicated exp41-generator runs with restricted "
                "verb inventories (exp42_c2_datagen.py): train on the "
                "A-inventory dataset, evaluate on the B-inventory "
                "dataset's test stories via the fixed kind-matched "
                "bijection (c2_bijection); pass = acc <= 0.55. The frozen "
                "main dataset is untouched.",
    }


def c2_eval_disjoint(model, compiled_b_stories, verb_inventory):
    """Evaluate a model whose blocks were trained on verbset-A stories on
    verbset-B stories via the fixed bijection. Pass = acc <= chance+5pp."""
    vmap = c2_bijection(verb_inventory)
    acc = _accuracy(model, compiled_b_stories, vmap)
    return {"control": "C2 informative-null", "verb_map": vmap,
            "accuracy": acc, "n_items": len(compiled_b_stories),
            "passed": bool(acc <= 0.55)}


# ----------------------------------------------------------------- C3
def c3_order_invariance(struct_models, seq_model, item, angles):
    """`struct_models`: dict name -> model consuming CompiledStory (quantum
    arms and/or B1). Returns machine-precision deltas for structured
    models and the input/behavior delta for the sequence baseline."""
    flipped = comp.rerender_sentences(item, flip=True)
    cs = comp.compile_story(item, angles, with_tokens=True)
    cs_f = comp.compile_story(flipped, angles, with_tokens=True)

    out = {"control": "C3 order-invariance", "item_id": item["id"],
           "n_sentences_rerendered":
               sum(1 for a, b in zip(item["order_flags"],
                                     flipped["order_flags"]) if a != b)}
    for name, m in struct_models.items():
        with torch.no_grad():
            v1 = m.story_values(cs)
            v2 = m.story_values(cs_f)
        delta = (v1 - v2).abs().max().item()
        out[name] = {"max_abs_value_delta": delta,
                     "machine_precision_identical": bool(delta == 0.0)}
    if seq_model is not None:
        tokens_differ = cs.tokens != cs_f.tokens
        p1 = seq_model.predict(cs)
        p2 = seq_model.predict(cs_f, tokens=cs_f.tokens)
        out["B2"] = {"input_tokens_differ": bool(tokens_differ),
                     "prediction_changed": bool(p1 != p2)}
    return out


def c3_b2_behavioral_delta(seq_model, items, angles):
    """Fraction of items where SVO<->VSO re-rendering flips B2's
    prediction (post-training measurement)."""
    flips = 0
    for it in items:
        cs = comp.compile_story(it, angles, with_tokens=True)
        cs_f = comp.compile_story(comp.rerender_sentences(it, flip=True),
                                  angles, with_tokens=True)
        if seq_model.predict(cs) != seq_model.predict(cs_f, tokens=cs_f.tokens):
            flips += 1
    return {"control": "C3 B2 behavioral delta",
            "n_items": len(items), "prediction_flip_fraction":
                flips / len(items) if items else None}


# ----------------------------------------------------------------- C4
def c4_sentence_scramble(model, compiled_items, seed=4444):
    """Accuracy on order-dependent items, original vs event-scrambled.
    Pass criterion (doc: 'must destroy accuracy'): scrambled accuracy on
    order-dependent items falls to <= 0.55."""
    rng = random.Random(seed)
    ordered = [cs for cs in compiled_items if cs.order_dependent]
    if not ordered:
        return {"control": "C4 sentence-scramble", "n_items": 0}
    acc_orig = _accuracy(model, ordered)
    scrambled = []
    for cs in ordered:
        ev = cs.events[:]
        while len(ev) > 1:
            rng.shuffle(ev)
            if ev != cs.events:
                break
        scrambled.append(replace(cs, events=ev))
    acc_scr = _accuracy(model, scrambled)
    return {"control": "C4 sentence-scramble",
            "n_order_dependent_items": len(ordered),
            "accuracy_original": acc_orig,
            "accuracy_scrambled": acc_scr,
            "destroyed": bool(acc_scr <= 0.55)}
