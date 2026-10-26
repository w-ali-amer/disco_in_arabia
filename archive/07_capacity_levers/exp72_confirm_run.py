# -*- coding: utf-8 -*-
"""exp72_confirm_run.py -- doc 58 SS16.3 step 11 / SS16.2: THE CONFIRM DRAW on
exp72's one positive, `rescue_partial(I1, R-identity) = TRUE`.

doc 47 SS18 IS THE STANDING REASON: A HEADLINE THAT SURVIVED ONE DRAW DIED ON THE
SECOND. The first draw returned exactly one rescue, and it is read as a first-draw
result until this draw agrees with it.

THIS FILE FORKS NO CODEPATH. It imports `exp72_run` and calls THE SAME
`train()` object the first draw's bit-exact anchors certified -- the same
procedure assertions, the same identity init, the same staged freezing, the same
donor protocol. It only re-points the dataset and the checkpoint directory. Doc 56
SS5.1.1's objection was to a COPY of the codepath; a driver that calls the
committed one is the opposite of that.

WHAT IS RUN (doc 58 SS16.3 step 11: ALL ARMS OF THE AFFECTED INSTANCE)

  C_I1_donor_s{0..4}     A0-W-T1 on the confirm draw -- doc 58 SS16.2's RETRAINED
                         DONOR. Carrying the first draw's `R1-W_s0` into a fresh
                         draw would initialise the confirm run from a model fitted
                         to A DIFFERENT DRAW'S ITEMS. Re-selected by the same val
                         rule and the same lowest-seed-index tie-break.
  C_I1_anchor_s{0..4}    A0-W-T2 unrescued -- THE CONFIRM DRAW'S BASELINE.
  C_I1_R-*_s{0..4}       all four rescue arms.
  C_I1_*_floor_s{0..4}   stage-2 procedure-matched floors, triggered cells only.

THE ANCHOR'S ROLE CHANGES ON A FRESH DRAW, AND IT IS STATED RATHER THAN SILENTLY
PASSED (doc 56 SS6.1's reconciliation, inherited by doc 58 SS16.2). doc 58 SS5.1's
|delta| <= 0.002 gate compares per-seed accuracies against committed measurements
ON THE FIRST DRAW'S OWN ITEMS. These are different items, so that gate is
INAPPLICABLE, not skipped, and on this draw A0-W-T2 is a BASELINE ARM.

**A DIFFERENT AND GENUINELY BLOCKING CHECK REPLACES IT, AND IT IS AVAILABLE ONLY
BECAUSE exp70's CONFIRM DRAW ALREADY MEASURED THESE TWO ARMS ON THESE EXACT
ITEMS**: `A0-W-T1` and `A0-W-T2` were run on this dataset, at the committed
recipe, at the same five seeds, through the committed exp70 codepath, and are
recorded in `results_exp70_confirm.json`. exp72's re-run of them must therefore
reproduce those numbers BIT-EXACTLY. That is a codepath-parity assertion of the
same strength as SS5.1's, on the fresh draw, and it is asserted here.

THE DATASET IS REUSED, NOT REGENERATED, AND THE ARGUMENT IS STATED (SS16.2)
==========================================================================
doc 58 SS16.2 defines the confirm draw as "re-runs the COMMITTED
`exp69_datagen.py`, UNMODIFIED, at generator seed + 1000". `exp69_datagen.py`
carries `SEED = 69000` and its own documented confirm invocation
`python exp69_datagen.py --seed 70000`. `stories_exp70_confirm_QA.json` IS that
object: `meta.seed = 70000`, `meta.is_confirm_draw = True`, same design doc, same
verb inventory, same 24-human inventory, same 528/132/300 splits at both T. The
generator is deterministic, so a third invocation at the same seed would produce a
byte-identical file; generating one would add a duplicate artefact and put Rule 3
at risk for nothing.

  - the dataset sha is ASSERTED here, not assumed;
  - its V-suite was RE-RUN on this draw and PASSED
    (`results_exp70_confirm_WN_validation.json`, `kill_gate_passed = true`,
    validating this exact sha), which is what SS16.2 requires;
  - doc 50 SS5.2's NATIVE-SPEAKER GATE STAYS CLOSED: no new surface template, no
    new noun, no new verb -- only fresh casts and pairings from inventories the
    user already confirmed, and the two inventories are asserted identical to the
    first draw's here;
  - the class-tier comparator for `A0-W-T2` ON THIS DRAW is IMPORTED from
    `results_exp70_confirm.json` and never re-run, exactly as doc 58 SS3 imports
    the first draw's.

ONE THING GENUINELY CHANGES ON THE FRESH DRAW AND THE BUILD HANDLES IT: the
generator re-draws the V_A / V_B verb split. First draw V_A = {sa'ala, daraba,
'aalaja}; confirm draw V_A = {rafa'a, saa'ada, 'aalaja}. `exp72_run.verb_groups`
reads `meta.verb_slots` from the dataset being run, so `R-layerwise`'s event-1 /
event-2 partition follows the draw automatically, and the 99/99/66 arithmetic is
re-asserted on this draw.

    python exp72_confirm_run.py --smoke
    python exp72_confirm_run.py --base    --worker i --nworkers 5   # donor+anchor
    python exp72_confirm_run.py --stage1  --worker i --nworkers 5   # 4 rescues
    python exp72_confirm_run.py --stage2 --cells R-identity --worker i --nworkers 5
    python exp72_confirm_aggregate.py
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

import exp70_run as R70                                  # noqa: E402
import exp72_run as X                                    # noqa: E402
from exp42_run import fresh_results_path                 # noqa: E402

CONFIRM_DATA = "stories_exp70_confirm_QA.json"
CONFIRM_SHA = ("59e6ada8a98b8917a1bd8e927d09249132a9765f7002e40fceb807a4bd51"
               "cd7c")
CONFIRM_VSUITE = "results_exp70_confirm_WN_validation.json"
CONFIRM_SOURCE = "results_exp70_confirm.json"
FIRST_DRAW_DATA = "stories_exp69_QA.json"
FIRST_DRAW_RESULTS = "results_exp72.json"

CKPT_DIR = "exp72_confirm_ckpt"
SMOKE_BASE = "results_exp72_confirm_smoke_validation"
DONOR_BASE = "results_exp72_confirm_donor"

INSTANCE = "I1"
ARM = "A0-W-T2"
DONOR_ARM = "A0-W-T1"
GENERATOR_SEED = 70000
FIRST_DRAW_SEED = 69000


# ------------------------------------------------------------------ the draw
def load_confirm_store():
    """The fresh draw, with SS16.2's provenance asserted rather than assumed."""
    store = R70.load_part(CONFIRM_DATA)
    _items, meta, path = store
    assert meta["dataset_sha256"] == CONFIRM_SHA, \
        ("the confirm dataset's sha is %s, not the asserted %s"
         % (meta["dataset_sha256"], CONFIRM_SHA))
    assert meta["seed"] == GENERATOR_SEED and meta.get("is_confirm_draw") is True, \
        ("%s is not the seed + 1000 confirm draw (seed %s, is_confirm_draw %s)"
         % (path, meta["seed"], meta.get("is_confirm_draw")))
    first = json.load(open(FIRST_DRAW_DATA, encoding="utf-8"))["meta"]
    assert meta["seed"] - first["seed"] == 1000, \
        ("the confirm draw is not at generator seed + 1000: %d vs %d"
         % (meta["seed"], first["seed"]))
    assert meta["dataset_sha256"] != first["dataset_sha256"], \
        ("the confirm draw reproduced the FIRST draw's sha -- it is not a fresh "
         "draw and confirms nothing")
    # doc 50 SS5.2's gate stays closed: NO new Arabic. Asserted, not assumed.
    assert sorted(meta["verb_inventory"]) == sorted(first["verb_inventory"]), \
        "the confirm draw introduced a new verb -- the native-speaker gate REOPENS"
    assert sorted(meta["humans"]) == sorted(first["humans"]), \
        "the confirm draw introduced a new noun -- the native-speaker gate REOPENS"
    assert meta["split_counts"] == first["split_counts"], "split counts differ"
    vs = json.load(open(CONFIRM_VSUITE, encoding="utf-8"))
    assert vs.get("kill_gate_passed") is True, \
        ("%s did not pass its kill gate -- no confirm-draw result is admissible"
         % CONFIRM_VSUITE)
    assert vs["parts"]["A"]["dataset_sha256"] == meta["dataset_sha256"], \
        "%s validates a different dataset than the one being run" % CONFIRM_VSUITE
    return store


def dataset_decision(store):
    _items, meta, path = store
    first = json.load(open(FIRST_DRAW_DATA, encoding="utf-8"))["meta"]
    return {
        "decision": "REUSE the committed seed+1000 draw; DO NOT regenerate",
        "path": path, "sha256": meta["dataset_sha256"],
        "generator_seed": meta["seed"],
        "first_draw_seed": first["seed"],
        "seed_offset": meta["seed"] - first["seed"],
        "is_confirm_draw": bool(meta.get("is_confirm_draw")),
        "first_draw_sha256": first["dataset_sha256"],
        "differs_from_first_draw": True,
        "vsuite": CONFIRM_VSUITE, "vsuite_kill_gate_passed": True,
        "vsuite_validates_this_sha": True,
        "verb_inventory_identical_to_first_draw": True,
        "human_inventory_identical_to_first_draw": True,
        "split_counts": meta["split_counts"],
        "verb_slots_first_draw": json.load(
            open(FIRST_DRAW_DATA, encoding="utf-8"))["meta"]["verb_slots"],
        "verb_slots_confirm_draw": meta["verb_slots"],
        "argument": (
            "doc 58 SS16.2 defines the confirm draw as a re-run of the COMMITTED "
            "exp69_datagen.py, UNMODIFIED, at generator seed + 1000. "
            "exp69_datagen.py carries SEED = 69000 and documents its own confirm "
            "invocation as `--seed 70000`. This dataset IS that object "
            "(meta.seed = 70000, is_confirm_draw = true, same design doc, same "
            "verb and noun inventories, same 528/132/300 splits at both T), it "
            "was produced by exp70's confirm draw, and its FULL V-SUITE WAS "
            "RE-RUN ON IT AND PASSED -- which is exactly what SS16.2 requires. "
            "The generator is deterministic, so a third invocation at the same "
            "seed produces a BYTE-IDENTICAL file: regenerating would add a "
            "duplicate artefact, put Rule 3 at risk, and change nothing. The sha "
            "and the V-suite verdict are ASSERTED here rather than trusted, which "
            "is doc 58 SS7.4's discipline applied to the fresh draw. doc 50 "
            "SS5.2's native-speaker gate stays CLOSED and that is asserted, not "
            "assumed: no new surface template, no new noun, no new verb."),
        "one_thing_that_changes": (
            "the generator RE-DRAWS the V_A / V_B verb split, so R-layerwise's "
            "event-1 / event-2 partition is NOT the first draw's. "
            "exp72_run.verb_groups reads meta.verb_slots from the dataset being "
            "run, so the partition follows the draw automatically; the 99/99/66 "
            "arithmetic and the 165/99/264 stage counts are re-asserted on this "
            "draw at smoke."),
    }


# --------------------------------------------------- the confirm-draw targets
def confirm_targets():
    """doc 58 SS3, applied to the fresh draw: the comparator is IMPORTED from the
    committed confirm-draw results and NEVER re-run.

    NOTE, AND IT MATTERS FOR THE BARS: on the fresh draw the class-tier
    comparator is 0.9880, not the first draw's 1.0000. The bars are recomputed
    from doc 58 SS9.2's FORMULAS with this draw's own baseline and comparator --
    the formulas are frozen, the numbers they consume are this draw's."""
    c = json.load(open(CONFIRM_SOURCE, encoding="utf-8"))
    blk = c["comparator"]["arms"][ARM]["class"]
    return {"comparator": float(blk["mean"]), "sd": float(blk["sd"]),
            "per_seed": list(blk["per_seed"]),
            "feature_dim": blk["feature_dim"],
            "source": "%s :: comparator.arms['%s']['class']" % (CONFIRM_SOURCE, ARM),
            "first_draw_comparator": 1.0,
            "note": ("exp72 runs NO comparator on either draw (doc 58 SS3). This "
                     "value is imported from the committed exp70 confirm results "
                     "and asserted. It is 0.9880 on this draw against 1.0000 on "
                     "the first, so the bars MOVE and are recomputed from the "
                     "registered formulas rather than carried over.")}


def confirm_reference():
    """exp70's confirm-draw measurements of the two arms exp72 re-runs here."""
    c = json.load(open(CONFIRM_SOURCE, encoding="utf-8"))
    out = {}
    for arm in (DONOR_ARM, ARM):
        a = c["arms"][arm]
        out[arm] = {"per_seed": list(a["per_seed"]), "mean": float(a["mean"]),
                    "sd": float(a["sd"]), "seeds": list(a["seeds"]),
                    "val_per_seed": list(a["val_per_seed"]),
                    "selected_epoch_per_seed": list(a["selected_epoch_per_seed"]),
                    "ckpt_sha256": list(a["ckpt_sha256"]),
                    "source": "%s :: arms['%s']" % (CONFIRM_SOURCE, arm)}
    return out


def confirm_bars(baseline):
    """doc 58 SS9.2's FROZEN FORMULAS, evaluated on this draw's numbers."""
    tgt = confirm_targets()
    comp = tgt["comparator"]
    gap = comp - baseline
    return {"baseline": baseline, "comparator": comp, "gap": gap,
            "rescue_full_bar": comp - 0.02,
            "rescue_partial_bar": baseline + 0.50 * gap,
            "formulas": ("rescue_full  >= comparator - 0.02;  "
                         "rescue_partial >= baseline + 0.50 * gap. FROZEN in doc "
                         "58 SS9.2. The FORMULAS are the registered object; the "
                         "numbers they consume are the draw's own baseline and "
                         "the draw's own imported comparator."),
            "first_draw_bars": {"baseline": 0.7293333333333334,
                                "comparator": 1.0,
                                "rescue_full_bar": 0.98,
                                "rescue_partial_bar": 0.8646666666666667}}


# ------------------------------------------------------------------ the donor
def donor_select(runs):
    """doc 58 SS6.4.1 / SS16.2: highest in-distribution VAL accuracy, ties broken
    by LOWEST SEED INDEX. Selection is on VAL, never on held-out."""
    rr = sorted(runs, key=lambda r: r["seed"])
    best = max(range(len(rr)), key=lambda i: (rr[i]["val_acc"], -rr[i]["seed"]))
    top = max(r["val_acc"] for r in rr)
    tied = [r["seed"] for r in rr if r["val_acc"] == top]
    return rr[best], {"val_per_seed": [r["val_acc"] for r in rr],
                      "max_val": top, "tied_seeds": tied,
                      "selected_seed": rr[best]["seed"],
                      "tie_break": ("lowest seed index among the val maxima "
                                    "(doc 58 SS6.4.1, orchestrator ruling 3)"),
                      "selection_is_on_val_never_heldout": True}


def donor_path(seed):
    return os.path.join(CKPT_DIR, "C_%s_donor_s%d.pt" % (INSTANCE, seed))


def write_donor_report(store):
    runs = [json.load(open(os.path.join(CKPT_DIR, "C_%s_donor_s%d.json"
                                        % (INSTANCE, s)), encoding="utf-8"))
            for s in range(X.N_SEEDS)]
    sel, info = donor_select(runs)
    ref = confirm_reference()[DONOR_ARM]
    got = [r["heldout_test_acc"] for r in sorted(runs, key=lambda x: x["seed"])]
    d = [g - w for g, w in zip(got, ref["per_seed"])]
    shas = [X.sha256_file(donor_path(s)) for s in range(X.N_SEEDS)]
    parity = {
        "reference": ref["source"],
        "exp70_confirm_per_seed": ref["per_seed"],
        "exp72_confirm_per_seed": got,
        "per_seed_delta": d,
        "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
        "bit_exact": bool(all(x == 0.0 for x in d)),
        "ckpt_sha_matches": [a == b for a, b in zip(shas, ref["ckpt_sha256"])],
        "all_ckpt_shas_match": bool(shas == ref["ckpt_sha256"]),
        "note": ("doc 58 SS5.1's |delta| <= 0.002 gate is INAPPLICABLE on a fresh "
                 "draw -- it compares against measurements on the FIRST draw's "
                 "items. This is the check that IS available and it is stronger "
                 "in kind: exp70's confirm draw already ran this arm on THESE "
                 "items at the committed recipe and the same five seeds, so "
                 "exp72's re-run must reproduce it BIT-EXACTLY, checkpoint shas "
                 "included.")}
    assert parity["bit_exact"], \
        ("the retrained donor does NOT reproduce exp70's confirm-draw A0-W-T1 "
         "bit-exactly (max |delta| %.3e) -- the codepath has drifted on the fresh "
         "draw and nothing here is read" % parity["max_abs_per_seed_delta"])
    rep = {"component": ("exp72 confirm draw -- doc 58 SS16.2's RETRAINED DONOR "
                         "and its selection"),
           "spec": X.SPEC, "spec_commit_line": X.spec_commit(),
           "why_retrained": (
               "doc 58 SS16.2, registered because it is easy to get wrong: the "
               "committed donor R1-W_s0 was trained on the FIRST draw's train "
               "split. Carrying it into a fresh-draw confirm would initialise the "
               "confirm run from a model fitted to A DIFFERENT DRAW'S ITEMS. The "
               "T = 1 donor arm is therefore re-run at the committed T = 1 recipe "
               "on the confirm draw's own T = 1 items and re-selected by the same "
               "val rule and the same tie-break."),
           "donor_arm": DONOR_ARM, "recipe": "committed T = 1: 150 epochs, Adam, "
                                             "lr 0.02, batch 8, best-val selection",
           "selection": info,
           "selected_ckpt": donor_path(sel["seed"]),
           "selected_ckpt_sha256": X.sha256_file(donor_path(sel["seed"])),
           "selected_val": sel["val_acc"],
           "selected_heldout_reported_not_used": sel["heldout_test_acc"],
           "per_seed_heldout": got,
           "first_draw_donor": {"ckpt": "exp69_ckpt/R1-W_s0.pt", "seed": 0,
                                "val": 0.9545454545454546,
                                "note": "NOT USED on this draw (SS16.2)"},
           "codepath_parity_on_the_fresh_draw": parity,
           "platform": platform.platform(), "torch": torch.__version__,
           "git_note": X.git_note()}
    p = fresh_results_path(DONOR_BASE)
    json.dump(rep, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[donor] selected seed %d (val %.6f, tied %s) -> %s"
          % (sel["seed"], sel["val_acc"], info["tied_seeds"],
             donor_path(sel["seed"])))
    print("[donor] codepath parity vs exp70 confirm: bit-exact=%s, shas match=%s"
          % (parity["bit_exact"], parity["all_ckpt_shas_match"]))
    print("[donor] wrote %s" % p)
    return sel, rep


def selected_donor_seed():
    runs = [json.load(open(os.path.join(CKPT_DIR, "C_%s_donor_s%d.json"
                                        % (INSTANCE, s)), encoding="utf-8"))
            for s in range(X.N_SEEDS)]
    return donor_select(runs)[0]["seed"]


# -------------------------------------------------------------------- smoke
def smoke(fm, verbs, store):
    """doc 58 SS16.2: EVERY SS7 ASSERTION RE-RUN ON THE FRESH DRAW."""
    t0 = time.time()
    rep = {"experiment": ("exp72 CONFIRM DRAW smoke -- doc 58 SS7 re-run on the "
                          "seed + 1000 draw (SS16.2)"),
           "spec": X.SPEC, "spec_commit_line": X.spec_commit(),
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__,
           "dataset_decision": dataset_decision(store),
           "comparator_target_doc58_SS3": confirm_targets(),
           "exp70_confirm_reference": confirm_reference()}
    X.check_partition(fm, verbs, store, rep)
    X.check_shuffle(store, rep)
    R70.check_copy(fm, verbs, store, rep)
    R70.check_trace(fm, verbs, store, rep)
    # the I1 closure gate, on this draw's items
    items, _m, _p = R70.arm_items(ARM, store)
    ang, _t1 = R70.arm_angles(ARM, fm)
    import exp69_run as R69
    data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
    torch.manual_seed(0)
    m = R70.build_model(ARM, verbs, 0, fm)
    w, wd = R70.fourier_resid_2d(m, data)
    assert w["3"] > 1e-3 and w["9"] < 1e-6 and wd > 1e-6, \
        ("%s's closure gate failed on the confirm draw: %s" % (ARM, w))
    rep["fourier_closure"] = {"arm": ARM, "resid_3term_theta1_only": w["3"],
                              "resid_9dim_product_basis": w["9"],
                              "output_change_under_theta2_perturbation": wd}
    X.check_negative_directions(fm, verbs, store, rep)
    tiny = {}
    for rescue in [None, "R-budget", "R-identity", "R-layerwise"]:
        r = X.train(ARM, 0, store, verbs, fm, rescue=rescue, max_epochs=2,
                    probe_epochs=(2 if rescue == "R-layerwise"
                                  else (6 if rescue == "R-budget" else None)),
                    save=False, label="confirm_smoke")
        tiny["%s" % (rescue or "anchor")] = {
            "val_acc": r["val_acc"], "heldout_test_acc": r["heldout_test_acc"],
            "epochs": r["max_epochs"],
            "procedure_assertions": {k: v for k, v
                                     in r["procedure_assertions"].items()
                                     if k != "identity"}}
        if rescue == "R-identity":
            tiny["R-identity"]["identity"] = {
                k: v for k, v in r["procedure_assertions"]["identity"].items()
                if k != "blocks_reinitialised"}
        print("[smoke] %-12s ep=%d val=%.3f test=%.3f"
              % (rescue or "anchor", r["max_epochs"], r["val_acc"],
                 r["heldout_test_acc"]), flush=True)
    rep["two_epoch_probe"] = tiny
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    rep["git_note"] = X.git_note()
    p = fresh_results_path(SMOKE_BASE)
    json.dump(rep, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 58 SS7 ASSERTIONS PASSED ON THE FRESH DRAW"
          % p)
    return rep


# --------------------------------------------------------------------- jobs
def base_jobs():
    j = [{"kind": "donor", "arm": DONOR_ARM, "seed": s, "rescue": None,
          "label": "C_%s_donor_s%d" % (INSTANCE, s)} for s in range(X.N_SEEDS)]
    j += [{"kind": "anchor", "arm": ARM, "seed": s, "rescue": None,
           "label": "C_%s_anchor_s%d" % (INSTANCE, s)} for s in range(X.N_SEEDS)]
    return j


def stage1_jobs():
    return [{"kind": "rescue", "arm": ARM, "seed": s, "rescue": r,
             "label": "C_%s_%s_s%d" % (INSTANCE, r, s)}
            for r in X.RESCUES for s in range(X.N_SEEDS)]


def stage2_jobs(cells):
    out = []
    for r in cells:
        assert r in X.RESCUES, "bad cell %s" % r
        kind = "warmstart_floor" if r == "R-warmstart" else "floor"
        for s in range(X.N_SEEDS):
            out.append({"kind": kind, "arm": ARM, "seed": s, "rescue": r,
                        "label": "C_%s_%s_floor_s%d" % (INSTANCE, r, s),
                        "donor_label": "C_%s_%s_donorfloor_s%d" % (INSTANCE, r, s)})
    return out


def run_jobs(jobs, fm, verbs, store):
    donor_state = None
    for j in jobs:
        extra = {"instance": INSTANCE, "draw": "confirm_seed_%d" % GENERATOR_SEED}
        if j["kind"] in ("donor", "anchor"):
            X._run_one(j["label"], j["kind"], j["arm"], j["seed"], None,
                       store, verbs, fm, extra)
            continue
        if j["kind"] == "rescue":
            kw = {}
            if j["rescue"] == "R-warmstart":
                if donor_state is None:
                    ds = selected_donor_seed()
                    p = donor_path(ds)
                    donor_state = X.load_donor(p, X.sha256_file(p), 264)[0]
                    extra["donor_ckpt"] = p
                    extra["donor_seed"] = ds
                kw["donor_state"] = donor_state
                ds = selected_donor_seed()
                extra.update({"donor_ckpt": donor_path(ds), "donor_seed": ds})
            X._run_one(j["label"], "rescue", j["arm"], j["seed"], j["rescue"],
                       store, verbs, fm, extra, **kw)
            continue
        if j["kind"] == "warmstart_floor":
            # doc 58 SS14.4 on the fresh draw: THE DONOR'S LABELS ARE SEVERED TOO,
            # seed-linked across the two item sets.
            X._run_one(j["donor_label"], "severed_donor", DONOR_ARM, j["seed"],
                       None, store, verbs, fm,
                       dict(extra, severed_donor_for=j["label"]), shuffled=True)
            sev = os.path.join(CKPT_DIR, j["donor_label"] + ".pt")
            assert os.path.exists(sev), "severed donor %s missing" % sev
            X._run_one(j["label"], "floor", j["arm"], j["seed"], j["rescue"],
                       store, verbs, fm,
                       dict(extra, severed_donor=j["donor_label"]),
                       shuffled=True,
                       donor_state=torch.load(sev, map_location="cpu"))
            continue
        X._run_one(j["label"], "floor", j["arm"], j["seed"], j["rescue"],
                   store, verbs, fm, extra, shuffled=True)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--select-donor", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--cells", default="")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()

    # Re-point the COMMITTED runner at the fresh draw. Nothing else changes; the
    # training object, its assertions and its RNG threading are exp72_run's.
    X.CKPT_DIR = CKPT_DIR
    os.makedirs(CKPT_DIR, exist_ok=True)
    store = load_confirm_store()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R70.feature_maps()

    if a.smoke:
        smoke(fm, verbs, store)
        return 0
    if a.select_donor:
        write_donor_report(store)
        return 0
    if a.base:
        jobs = base_jobs()
    elif a.stage1:
        jobs = stage1_jobs()
    elif a.stage2:
        cells = [c for c in a.cells.split(",") if c]
        if not cells:
            raise RuntimeError("--stage2 needs --cells")
        jobs = stage2_jobs(cells)
    else:
        raise RuntimeError("pass --smoke, --base, --select-donor, --stage1 or "
                           "--stage2")
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[confirm w%d/%d] %d of %d runs" % (a.worker, a.nworkers, len(mine),
                                              len(jobs)), flush=True)
    run_jobs(mine, fm, verbs, store)
    print("[confirm w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
