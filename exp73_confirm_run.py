"""exp73 CONFIRM driver -- doc 61 SS21.3, ordered 2026-09-01.

Runs the doc 61 SS17.2 confirm draw for the two fired positives
(amplitude_hurts_T2, init_helps_A3_T2): the four A3 arms, their four
S-shuffle floors re-drawn matched to the fresh draw, and the anchor
A0-W-T2 as the same-draw committed-register reference -- 5 seeds each,
on `stories_exp70_confirm_QA.json` (the committed `exp69_datagen.py`,
UNMODIFIED, at generator seed 70000 -- the same file exp72's confirm
validated, sha re-asserted below).

WHAT THIS DRIVER PATCHES, EXHAUSTIVELY (doc 61 SS21.3 licenses exactly
this and nothing else):
  1. the dataset binding (`R69.DATA_A`), so the committed runner's
     SS17.1 frozen-Part-A guard admits the ordered confirm draw;
  2. the registered-sha constant (`DATASET_SHA_DOC61` -> CONFIRM_SHA),
     so the SS7.3 sha assertion checks the CONFIRM draw's identity
     instead of the first draw's;
  3. the V-suite binding (`VSUITE` -> the confirm draw's own V-suite,
     `results_exp70_confirm_WN_validation.json`, kill gate passed,
     produced for this exact sha by exp72's confirm -- inherited on the
     same byte-identity argument doc 61 SS7.1 used for exp69's);
  4. `check_dataset` -> a confirm variant below that performs the SAME
     checks (meta-sha assertion, V-suite kill gate) against the confirm
     bindings, and additionally asserts the draw is FRESH (sha differs
     from the first draw's).
Training, evaluation, arms, floors, features, instrument provenance and
every other SS7 assertion are `exp73_run.py`'s committed code paths,
unforked. The anchor REPLICATION report (doc 61 SS5.3, --anchor-report)
is not run here: it compares against the first draw's committed
accuracies and is meaningless on a fresh draw; it was passed bit-exact
on the frozen draw by the stage aggregator today (2026-09-01). A0-W-T2
runs as an ordinary arm to supply the same-draw reference of SS21.4
bar 2.
"""
import json
import sys

CONFIRM_DATA = "stories_exp70_confirm_QA.json"
CONFIRM_CKPT = "exp73_confirm_ckpt"
CONFIRM_VSUITE = "results_exp70_confirm_WN_validation.json"
CONFIRM_SHA = ("59e6ada8a98b8917a1bd8e927d09249132a9765f7002e40fceb807a4bd51"
               "cd7c")
FIRST_DRAW_SHA = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                  "a6308c96")
ARMS = ",".join([
    "A3-T1-uniform", "A3-T1-identity", "A3-T2-uniform", "A3-T2-identity",
    "S-shuffle-A3-T1-uniform", "S-shuffle-A3-T1-identity",
    "S-shuffle-A3-T2-uniform", "S-shuffle-A3-T2-identity",
    "A0-W-T2",
])


def make_confirm_check_dataset(R):
    def confirm_check_dataset(store, rep):
        """doc 61 SS21.3's confirm variant of the SS7.3 dataset gate: the
        same assertions, against the CONFIRM draw's registered identity."""
        _items, meta, path = store
        assert meta["dataset_sha256"] == CONFIRM_SHA, \
            ("confirm dataset sha mismatch: %s vs the doc 61 SS21.3 "
             "registered %s" % (meta["dataset_sha256"], CONFIRM_SHA))
        assert meta["dataset_sha256"] != FIRST_DRAW_SHA, \
            "the confirm draw reproduced the FIRST draw -- it is not fresh"
        vs = json.load(open(CONFIRM_VSUITE, encoding="utf-8"))
        assert vs.get("kill_gate_passed") is True, \
            "the confirm draw's V-suite kill gate did not pass"
        rep["dataset"] = {
            "path": path, "sha256": meta["dataset_sha256"],
            "registered_sha_doc61_SS21_3": CONFIRM_SHA,
            "is_confirm_draw": True, "generator_seed": meta.get("seed"),
            "differs_from_first_draw": True,
            "vsuite_source": CONFIRM_VSUITE, "vsuite_kill_gate_passed": True,
            "vsuite_status": ("INHERITED from exp72's confirm on the strength "
                              "of byte identity (same sha) -- doc 61 SS21.3"),
        }
        return rep
    return confirm_check_dataset


def main():
    meta = json.load(open(CONFIRM_DATA, encoding="utf-8"))["meta"]
    assert meta["dataset_sha256"] == CONFIRM_SHA, \
        ("doc 61 SS21.3: confirm dataset sha mismatch -- got %r, STOP"
         % meta.get("dataset_sha256"))

    import exp73_run as R
    R.R69.DATA_A = CONFIRM_DATA
    R.DATASET_SHA_DOC61 = CONFIRM_SHA
    R.VSUITE = CONFIRM_VSUITE
    R.check_dataset = make_confirm_check_dataset(R)

    passthrough = [a for a in sys.argv[1:]]
    sys.argv = ([sys.argv[0], "--data", CONFIRM_DATA,
                 "--ckpt-dir", CONFIRM_CKPT,
                 "--smoke-out", "results_exp73_confirm_smoke.json",
                 "--anchor-out", "results_exp73_confirm_anchor.json",
                 "--vsuite", CONFIRM_VSUITE,
                 "--arms", ARMS] + passthrough)
    sys.exit(R.main())


if __name__ == "__main__":
    main()
