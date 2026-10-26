# -*- coding: utf-8 -*-
"""exp78a_run.py -- thin driver for exp78a: the band32 replication.

Registration: qnlp_private_docs/68_exp78a_band32_replication.md §2.

Patches ONLY: dataset binding, sha pin, output dirs -- per the doc 62
§24.3-lineage checklist (sha asserted not assumed, separate results file
results_exp78a.json, separate output dirs; no default-arg rebinding needed
as exp76_run uses no default-arg data paths).

Arms: band32_n2_T2, S-shuffle-band32_n2_T2 (5 seeds each, 10 runs).
Data: stories_exp78a_confirm_{cell}.json  (CONFIRM_SEED = 77000).
Checkpoints: exp78a_ckpt/

Patch order (load-bearing):
  1. exp76_datagen.STORY_FMT patched BEFORE exp76_run import so load_store()
     reads from confirm paths when it is called inside R.main().
  2. exp76_run.EXP76_VSUITE patched AFTER exp76_run import because
     EXP76_VSUITE = D76.VALIDATION_OUT binds at exp76_run module load time
     (D76.VALIDATION_OUT was not yet patched at that moment).
  3. Freshness assertions run before R.main() at every invocation.

FLAGGED AMBIGUITY: exp76_run.check_datasets makes no per-cell sha assertion
for trained cells (only exp69's frozen DATASET_SHA_DOC63 is hardcoded). The
driver-level freshness check below substitutes: sha != first_draw_sha,
seed == 77000, is_confirm_draw == True.  Resolved toward committed code,
flagged per doc 63 standing rule.

Usage:
  python exp78a_run.py --smoke                   # all SS8 gates, confirm draw
  python exp78a_run.py --worker i --nworkers 5   # train 2/10 runs per worker
"""
import sys
import json

# ----------------------------------------------------------- registered constants
CONFIRM_SEED   = 77000
CONFIRM_FMT    = "stories_exp78a_confirm_%s.json"
CONFIRM_VSUITE = "results_exp78a_WN_validation.json"
CONFIRM_CKPT   = "exp78a_ckpt"
CONFIRM_SMOKE  = "results_exp78a_smoke_validation.json"
CONFIRM_ANCHOR = "results_exp78a_anchor.json"
TRAIN_ARMS     = "band32_n2_T2,S-shuffle-band32_n2_T2"
CONFIRM_CELL   = "band32_n2_T2"

# sha of the committed first draw (stories_exp76_band32_n2_T2.json, seed 76000)
FIRST_DRAW_SHA = ("ebdf32a2deb7bc6211bd69e9c673fb341498c3156a2fcdddf5de5a481481ed20")
# sha of the confirm draw (stories_exp78a_confirm_band32_n2_T2.json, seed 77000)
CONFIRM_SHA    = ("128e86d5f96efdb8ea7ea2dc297d7ee6b6147958b423ae7b43c16c90a27c2510")

# ------------------------------------------- patch 1: STORY_FMT before R import
import exp76_datagen as D76

D76.STORY_FMT    = CONFIRM_FMT
D76.VALIDATION_OUT = CONFIRM_VSUITE   # belt-and-suspenders; R.EXP76_VSUITE below

# ------------------------------------------ import committed runner and patch 2
import exp76_run as R

# EXP76_VSUITE bound at exp76_run import from D76.VALIDATION_OUT (which was
# still the original "results_exp76_WN_validation.json" at that instant).
# Patch directly on the module so check_datasets() reads the confirm report.
R.EXP76_VSUITE = CONFIRM_VSUITE

# ---------------------------------------------------------------- freshness gate
# Doc 68 §2: sha asserted not assumed.  Runs at every invocation.
_cp = CONFIRM_FMT % CONFIRM_CELL
try:
    _blob = json.load(open(_cp, encoding="utf-8"))
except FileNotFoundError:
    raise RuntimeError(
        "Confirm dataset %s not found -- run `python exp78a_datagen.py` first"
        % _cp)
_meta = _blob["meta"]
assert _meta["dataset_sha256"] == CONFIRM_SHA, (
    "doc 68 §2: confirm sha mismatch at %s (%r)" % (_cp, _meta["dataset_sha256"]))
assert _meta["dataset_sha256"] != FIRST_DRAW_SHA, (
    "doc 68 §2: confirm draw reproduced the first draw sha -- not fresh")
assert _meta.get("seed") == CONFIRM_SEED, (
    "doc 68 §2: meta.seed is %r, expected %d" % (_meta.get("seed"), CONFIRM_SEED))
assert _meta.get("is_confirm_draw") is True, (
    "doc 68 §2: meta.is_confirm_draw is not True")

if __name__ == "__main__":
    # Bake confirm-scope redirects first; user's argv (--smoke / --worker i
    # --nworkers 5 / --anchor-report) are appended as passthrough.
    # --arms is baked so workers train only the two registered cells; smoke
    # ignores --arms (it probes ALL_ARMS directly) so baking is harmless there.
    passthrough = list(sys.argv[1:])
    sys.argv = (
        [sys.argv[0],
         "--arms",      TRAIN_ARMS,
         "--ckpt-dir",  CONFIRM_CKPT,
         "--smoke-out", CONFIRM_SMOKE,
         "--anchor-out", CONFIRM_ANCHOR]
        + passthrough
    )
    sys.exit(R.main())
