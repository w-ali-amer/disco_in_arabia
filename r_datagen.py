# -*- coding: utf-8 -*-
"""r_datagen.py -- doc 25 SSR.2 (frozen): regenerate the L2 rung and the exp43a
family on the REPAIRED generator (exp41_datagen.py, doc 65 SS6.3).

Registered procedure, implemented verbatim:
  * exp41_datagen.generate(seed, spec=...) with BOTH namespaces rebound -- the
    exp41_datagen copies AND the exp41_story_generator copies of VERBS /
    SWAP_VERBS -- in the original verb order:
        L2      the 6 swap verbs sorted      (exp43b_datagen.RUNG_VERBS, :41)
        family  VERB6 order                  (exp43a_family_datagen.VERB6, :42)
    (rebinding only gen.* would leave exp41_datagen's own VERBS at 15 verbs:
    a different model and a 15-verb inventory).
  * spec content unchanged, imported (not retyped) from the committed files:
        L2      exp43b_datagen.RUNGS["L2"]["spec"]   (:53-61; test split "test")
        family  exp43a_family_datagen.FAMILY_SPEC    (:37-50)
  * seeds L2r = 4312000, family-r = 4301000 (doc 25 SSR.1, x1000 precedent).
  * meta written HERE with verb_inventory = the rung's verbs (in rung order).
  * frozen-embedding requirement (exp43a_family_datagen.py:81-84) enforced on
    both datasets (both are compiled by exp42_compiler downstream).
  * new files only; Rule 3: refuses to overwrite.

Modes:
  python r_datagen.py                 generate + write + record
  python r_datagen.py --sha-only      print canonical shas at the registered
                                      seeds (g3 determinism probe; no files)
"""

import argparse
import json
import sys
from collections import Counter

import exp41_story_generator as gen
import exp41_datagen as G
import exp43b_datagen as B          # L2 spec + RUNG_VERBS (import only)
import exp43a_family_datagen as F   # FAMILY_SPEC + VERB6 (import only)
import r_common as rc

FULL_VERBS = dict(gen.VERBS)
FULL_SWAP = list(gen.SWAP_VERBS)
assert G.VERBS is gen.VERBS or G.VERBS == gen.VERBS

SPECS = {
    "L2r": {"seed": rc.L2R_SEED, "spec": B.RUNGS["L2"]["spec"],
            "verbs": list(B.RUNG_VERBS), "out": rc.L2R_DATA,
            "orig_seed": B.RUNGS["L2"]["seed"],
            "dry_run_sha_prefix": "b0d86f79"},
    "familyr": {"seed": rc.FAMR_SEED, "spec": F.FAMILY_SPEC,
                "verbs": list(F.VERB6), "out": rc.FAMR_DATA,
                "orig_seed": F.SEED,
                "dry_run_sha_prefix": "63270aab"},
}
RECORD = "r_datagen_record.json"


def build(which, seed=None, sabotage=None):
    """Generate one spec with both namespaces rebound; always restores."""
    cfg = SPECS[which]
    seed = cfg["seed"] if seed is None else seed
    V = {v: FULL_VERBS[v] for v in cfg["verbs"]}
    S = [v for v, c in V.items() if c["swap"]]
    gen.VERBS, gen.SWAP_VERBS = V, S
    G.VERBS, G.SWAP_VERBS = V, S
    try:
        items, holdout = G.generate(seed, sabotage=sabotage, spec=cfg["spec"])
    finally:
        gen.VERBS, gen.SWAP_VERBS = dict(FULL_VERBS), list(FULL_SWAP)
        G.VERBS, G.SWAP_VERBS = gen.VERBS, gen.SWAP_VERBS
    used = {e["verb"] for it in items for e in it["events"]}
    assert used <= set(cfg["verbs"]), "verb outside rung inventory: %s" % (
        used - set(cfg["verbs"]))
    return items, holdout


def make_meta(which, items, holdout, sha):
    cfg = SPECS[which]
    counts = Counter((it["split"], it["qtype"]) for it in items)
    return {
        "experiment": ("exp43b_L2r" if which == "L2r" else "exp43a_familyr"),
        "rung": "L2r" if which == "L2r" else None,
        "registration": rc.REGISTRATION,
        "generator": "exp41_datagen.generate (repaired, doc 65 SS6.3) with "
                     "exp41_datagen.* AND exp41_story_generator.* VERBS/"
                     "SWAP_VERBS rebound in the original order",
        "spec_source": ("exp43b_datagen.RUNGS['L2']['spec']" if which == "L2r"
                        else "exp43a_family_datagen.FAMILY_SPEC"),
        "seed": cfg["seed"],
        "dataset_sha256": sha,
        "n_items": len(items),
        "split_counts": dict(Counter(it["split"] for it in items)),
        "split_qtype_counts": {"%s/%s" % (s, q): n
                               for (s, q), n in sorted(counts.items())},
        "verb_inventory": {v: {"human_patients": FULL_VERBS[v]["human_patients"],
                               "swap": FULL_VERBS[v]["swap"],
                               "objects": FULL_VERBS[v]["objects"]}
                           for v in cfg["verbs"]},
        "verb_order": cfg["verbs"],
        "humans": gen.HUMANS, "objects": gen.OBJECTS,
        "holdout_verb_argument_pairs": sorted([list(p) for p in holdout]),
        "notes": [
            "CLEAN-DATA RE-DECISION dataset (doc 25 SSR); the August file is "
            "untouched and its booleans stand with the shortcut caveat",
            "items carry flip_idxs (a LIST: the repaired generator exchanges "
            "the core event AND the balancing event), not the August scalar "
            "flip_idx",
            ("L2r: Q0 fillers may not use the question verb at all "
             "(stricter than August), so the one-hop half also changes "
             "(doc 25 SSR.2 disclosure i)") if which == "L2r" else
            ("family-r: at K=4 (3 humans + 1 object) X is forced to be the "
             "OBJECT and v1 is an object-taking verb; 'same spec' is not "
             "'same family' (doc 25 SSR.2 disclosure ii)"),
        ],
    }


def frozen_embedding_check(items):
    import exp42_compiler as comp
    angles, _m, emb_sha = comp.load_embeddings()
    missing = sorted({r["name"] for it in items for r in it["referents"]
                      if r["name"] not in angles})
    assert not missing, "referents without frozen embeddings: %s" % missing
    return emb_sha


def disclosures(which, items):
    objs = set(gen.OBJECTS)
    out = {}
    if which == "L2r":
        q0 = [it for it in items if it["qtype"] == "Q0"]
        q1 = [it for it in items if it["qtype"] == "Q1"]
        out["Q0_items"] = len(q0)
        out["Q0_items_with_ANY_other_event_using_question_verb"] = sum(
            1 for it in q0 if sum(e["verb"] == it["question"]["v"]
                                  for e in it["events"]) > 1)
        out["Q1_items"] = len(q1)
        out["Q1_items_with_exactly_two_v2_events_(core+balancing)"] = sum(
            1 for it in q1 if sum(e["verb"] == it["question"]["v2"]
                                  for e in it["events"]) == 2)
        out["flip_idxs_length_by_qtype"] = {
            q: dict(Counter(len(it["flip_idxs"]) for it in items
                            if it["qtype"] == q)) for q in ("Q0", "Q1")}
        # the August L2 for contrast (same measure; read-only)
        with open("stories_exp43b_L2.json", encoding="utf-8") as f:
            aug = json.load(f)["items"]
        aq0 = [it for it in aug if it["qtype"] == "Q0"]
        out["AUGUST_L2_Q0_items_with_other_event_using_question_verb"] = sum(
            1 for it in aq0 if sum(e["verb"] == it["question"]["v"]
                                   for e in it["events"]) > 1)
        out["AUGUST_L2_Q0_items"] = len(aq0)
    else:
        xs_obj = sum(1 for it in items if it["question"]["x"] in objs)
        out["items"] = len(items)
        out["X_is_object"] = xs_obj
        out["v1_distribution"] = dict(Counter(it["question"]["v1"]
                                              for it in items))
        out["v2_distribution"] = dict(Counter(it["question"]["v2"]
                                              for it in items))
        out["K_T"] = dict(Counter("K%d_T%d" % (it["K"], it["T"])
                                  for it in items))
        with open("stories_exp43a_family.json", encoding="utf-8") as f:
            aug = json.load(f)["items"]
        out["AUGUST_family_X_is_object"] = sum(
            1 for it in aug if it["question"]["x"] in objs)
        out["AUGUST_family_items"] = len(aug)
    return out


def sha_only():
    out = {}
    for which in SPECS:
        items, _h = build(which)
        out[which] = rc.canonical_sha256(items)
    print(json.dumps(out, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha-only", action="store_true")
    args = ap.parse_args()
    if args.sha_only:
        return sha_only()

    for which in SPECS:
        rc.rule3(SPECS[which]["out"])
    rc.rule3(RECORD)

    record = {"registration": rc.REGISTRATION,
              "procedure": "doc 25 SSR.2 (frozen)", "datasets": {}}
    for which, cfg in SPECS.items():
        items, holdout = build(which)
        sha = rc.canonical_sha256(items)
        emb_sha = frozen_embedding_check(items)
        if which == "familyr":
            assert all(it["qtype"] == "Q1" and it["K"] == 4 and it["T"] == 5
                       for it in items), "family constraint violated"
            assert sum(it["split"] == "train" for it in items) >= 200
        else:
            assert {it["split"] for it in items} == {"train", "val", "test"}
        meta = make_meta(which, items, holdout, sha)
        meta["embeddings_sha256"] = emb_sha
        with open(rc.rule3(cfg["out"]), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "items": items}, f, ensure_ascii=False,
                      indent=1)
        # port-fidelity check: the dry run (doc 25 SSR.1) at the ORIGINAL
        # seed, in memory only; nothing is selected by it
        port_items, _ = build(which, seed=cfg["orig_seed"])
        port_sha = rc.canonical_sha256(port_items)
        record["datasets"][which] = {
            "file": cfg["out"], "seed": cfg["seed"],
            "canonical_items_sha256": sha,
            "file_sha256": rc.file_sha256(cfg["out"]),
            "n_items": len(items), "split_counts": meta["split_counts"],
            "split_qtype_counts": meta["split_qtype_counts"],
            "verb_order": cfg["verbs"],
            "embeddings_sha256": emb_sha,
            "disclosures": disclosures(which, items),
            "port_fidelity_check": {
                "original_seed": cfg["orig_seed"],
                "sha_at_original_seed_in_memory": port_sha,
                "doc25_R1_dry_run_prefix": cfg["dry_run_sha_prefix"],
                "matches_dry_run": port_sha.startswith(
                    cfg["dry_run_sha_prefix"])},
        }
        print("[r_datagen] %s -> %s  n=%d  sha=%s  port-check=%s"
              % (which, cfg["out"], len(items), sha,
                 record["datasets"][which]["port_fidelity_check"]
                 ["matches_dry_run"]), flush=True)
    record["flip_idxs_note"] = (
        "repaired-generator items carry `flip_idxs` (list). Old scripts that "
        "read the August scalar `flip_idx` would KeyError on these files: "
        "exp43a_analysis.py:234 (it['flip_idx']). None of the r_ runners "
        "(exp45a/45b/47/48/43b_phase1 via exp42_compiler) read either key.")
    with open(rc.rule3(RECORD), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    print("[r_datagen] record -> %s" % RECORD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
