# -*- coding: utf-8 -*-
"""exp42r_c2_datagen.py -- restricted-inventory datasets for control C2.

doc 22 SS6 C2, the INFORMATIVE NULL: "blocks trained on a disjoint verb
inventory, evaluated on the test verbs' stories -> must be <= chance +
5 pp." If blocks trained on verbset A solve verbset B's stories through a
kind-matched bijection, the blocks hold generic frame machinery rather than
verb-specific content, and a C1 pass would mean much less.

`stories_exp41r.json` mixes verbs freely, so it contains essentially no
verb-disjoint stories; C2 needs dedicated generator runs. Pre-training
generator work is explicitly licensed by doc 22 SS2.3, and the frozen
dataset of record is UNTOUCHED.

Outputs (revival-marked; the August artefacts stories_exp42_c2_trainA.json
and stories_exp42_c2_evalB.json are NEITHER written NOR read):
    stories_exp42r_c2_trainA.json   seed 4211, verbset A -> C2 training
    stories_exp42r_c2_evalB.json    seed 4212, verbset B -> C2 evaluation
Both are validated by `validate_exp41r_data.py` -- the REVIVAL validator
carrying doc 65 SS6.3's STRONG, GATING V5 -- never by the August
`validate_exp41_data.py`, whose order-blind V5 oracle doc 65 SS6.2 ruled
provably incapable of failing.

THE REBINDING TRAP, and why this file is not a copy of the August one.
`exp42_c2_datagen.py` rebinds only `exp41_story_generator.VERBS` /
`.SWAP_VERBS`. That sufficed in August because the August generator lived
in that module. The revival generator `exp41_datagen.py` does
`from exp41_story_generator import VERBS, SWAP_VERBS`, which BINDS ITS OWN
MODULE GLOBALS AT IMPORT TIME -- rebinding only the source module would
leave `exp41_datagen.VERBS` pointing at the full 15-verb inventory and
silently produce datasets that are not verb-disjoint at all, i.e. a C2
that cannot fail. BOTH modules are rebound here, and `verify_disjoint()`
re-reads the generated items and asserts inventory disjointness FROM THE
DATA rather than from the intent.

Feasibility, checked not assumed: the A/B split is
`exp42_controls.c2_verb_split` (deterministic, swap-balanced, imported
unmodified). The full inventory has 6 swap verbs, so each half gets
exactly 3, and `exp41_datagen.three_distinct_swap_verbs` needs exactly 3 --
feasible with ZERO slack. `--check` runs a reduced generation of both
halves and reports before anything is written.
"""

import argparse
import json
import sys
from collections import Counter

import exp41_story_generator as gen
import exp41_datagen as dg
import validate_exp41r_data as val
from exp42_controls import c2_verb_split

FULL_VERBS = dict(gen.VERBS)                              # snapshot

TRAIN_OUT = "stories_exp42r_c2_trainA.json"
TRAIN_MANIFEST = "stories_exp42r_c2_trainA_manifest.json"
EVAL_OUT = "stories_exp42r_c2_evalB.json"
EVAL_MANIFEST = "stories_exp42r_c2_evalB_manifest.json"
TRAIN_SEED = 4211
EVAL_SEED = 4212

# The C2 datasets use the DATASET OF RECORD'S OWN SPLIT SPEC, unreduced.
# A first build halved it -- C2 only needs a trainable A-corpus and a B test
# set -- and `validate_exp41r_data.MIN_TEST_PER_SPLIT = 500` correctly
# refused to validate the result. Rather than weaken the revival validator
# for a control dataset, the full spec is used: the V-suite that gates the
# dataset of record gates these too, at their own generator seeds.
C2_SPEC = dict(dg.SPLIT_SPEC)


def set_inventory(subset):
    """Rebind the verb inventory in BOTH modules (see the trap above)."""
    v = {k: FULL_VERBS[k] for k in subset}
    sw = [k for k, c in v.items() if c["swap"]]
    gen.VERBS, gen.SWAP_VERBS = v, sw
    dg.VERBS, dg.SWAP_VERBS = v, sw
    return v, sw


def restore_inventory():
    gen.VERBS = dict(FULL_VERBS)
    gen.SWAP_VERBS = [k for k, c in gen.VERBS.items() if c["swap"]]
    dg.VERBS = gen.VERBS
    dg.SWAP_VERBS = gen.SWAP_VERBS


def rebinding_selftest(subset):
    """Both modules must actually see the restricted inventory."""
    v, sw = set_inventory(subset)
    try:
        assert set(dg.VERBS) == set(subset), \
            "exp41_datagen.VERBS was NOT rebound -- the August rebinding " \
            "trick is insufficient for this module (see docstring)"
        assert set(gen.VERBS) == set(subset), "gen.VERBS not rebound"
        assert sorted(dg.SWAP_VERBS) == sorted(sw), "SWAP_VERBS not rebound"
        return {"subset": sorted(subset), "n_swap": len(sw),
                "swap_verbs": sorted(sw),
                "three_distinct_swap_verbs_feasible": len(sw) >= 3}
    finally:
        restore_inventory()


def verify_disjoint(items_a, items_b, va, vb):
    """Assert disjointness FROM THE DATA, not from the intent."""
    def used(items):
        s = set()
        for it in items:
            s.update(e["verb"] for e in it["events"])
            q = it["question"]
            s.update((q["v"],) if "v" in q else (q["v1"], q["v2"]))
        return s
    ua, ub = used(items_a), used(items_b)
    overlap = sorted(ua & ub)
    if overlap:
        raise RuntimeError(
            "C2 DATASETS ARE NOT VERB-DISJOINT: %s appear in both. The "
            "control cannot fail and must not be run." % overlap)
    stray_a = sorted(ua - set(va))
    stray_b = sorted(ub - set(vb))
    if stray_a or stray_b:
        raise RuntimeError("C2 inventory leak: A has %s outside verbset A, "
                           "B has %s outside verbset B" % (stray_a, stray_b))
    return {"verbs_used_A": sorted(ua), "verbs_used_B": sorted(ub),
            "disjoint": True}


def generate_restricted(subset, seed, out, manifest, label, spec):
    set_inventory(subset)
    try:
        items, holdout = dg.generate(seed, spec=spec)
        # write() MUST run inside the restricted scope: exp41_datagen.write
        # builds meta["verb_inventory"] from its MODULE GLOBAL `VERBS`, so
        # restoring the full inventory first stamps all 15 verbs into a
        # dataset that contains 7 or 8. The items were correct either way,
        # but the recorded inventory was not -- and exp42r_run's C2 job
        # asserts the manifest against exp42_controls.c2_verb_split, which
        # is exactly the assertion that caught this.
        sha, split_counts, counts = dg.write(items, holdout, seed, out,
                                             manifest)
    finally:
        restore_inventory()
    print("[c2r-datagen] %s -> %s: %d items, sha256=%s"
          % (label, out, len(items), sha), flush=True)
    for s in spec:
        print("   %-8s %5d  %s"
              % (s, split_counts[s],
                 "  ".join("%s=%d" % (q, counts[(s, q)])
                           for q in ("Q1", "Q2", "Q0"))), flush=True)
    return items, sha


def run_vsuite(path, results_base):
    """doc 65 SS6.3's STRONG V5 -- the revival validator, not the August one."""
    val.DATA, val.RESULTS_BASE = path, results_base
    print("[c2r-datagen] V-suite (strong V5) on %s -> %s*.json"
          % (path, results_base), flush=True)
    return val.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="feasibility probe only; writes nothing")
    ap.add_argument("--skip-vsuite", action="store_true")
    args = ap.parse_args()

    inv = {v: {"swap": c["swap"]} for v, c in FULL_VERBS.items()}
    va, vb = c2_verb_split(inv)
    print("[c2r-datagen] verbset A (train inventory): %s" % va)
    print("[c2r-datagen] verbset B (eval inventory):  %s" % vb)
    assert not (set(va) & set(vb)), "c2_verb_split produced overlap"

    ra, rb = rebinding_selftest(va), rebinding_selftest(vb)
    print("[c2r-datagen] rebinding self-test: A %s  B %s" % (ra, rb))
    for r, lab in ((ra, "A"), (rb, "B")):
        if not r["three_distinct_swap_verbs_feasible"]:
            sys.exit("[c2r-datagen] verbset %s has only %d swap verbs; "
                     "exp41_datagen.three_distinct_swap_verbs needs 3. C2 "
                     "is infeasible under this split and that is the report."
                     % (lab, r["n_swap"]))

    if args.check:
        tiny = {k: dict(v, Q1=6, Q2=3, Q0=(3 if k in ("train", "val") else 0))
                for k, v in C2_SPEC.items()}
        for subset, seed, lab in ((va, TRAIN_SEED, "A"), (vb, EVAL_SEED, "B")):
            set_inventory(subset)
            try:
                its, _h = dg.generate(seed, spec=tiny)
                print("[c2r-datagen] --check verbset %s: generated %d items OK"
                      % (lab, len(its)))
            finally:
                restore_inventory()
        print("[c2r-datagen] --check PASSED; nothing written.")
        return 0

    items_a, sha_a = generate_restricted(va, TRAIN_SEED, TRAIN_OUT,
                                         TRAIN_MANIFEST, "verbset A", C2_SPEC)
    items_b, sha_b = generate_restricted(vb, EVAL_SEED, EVAL_OUT,
                                         EVAL_MANIFEST, "verbset B", C2_SPEC)
    dj = verify_disjoint(items_a, items_b, va, vb)
    print("[c2r-datagen] verb-disjointness verified FROM THE DATA: %s" % dj)

    if not args.skip_vsuite:
        run_vsuite(TRAIN_OUT, "results_exp42r_c2_validation_trainA")
        run_vsuite(EVAL_OUT, "results_exp42r_c2_validation_evalB")

    print("[c2r-datagen] done. A sha=%s  B sha=%s" % (sha_a[:16], sha_b[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
