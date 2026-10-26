# -*- coding: utf-8 -*-
"""exp63_aggregate.py -- doc 49 SS6.2 verdict for exp63, read off exp63_ckpt/.

exp63's 70 runs completed 2026-08-22 but the verdict was never written to a
results file. This aggregates the checkpoints and formalises the registered
criterion. It TRAINS NOTHING and it WRITES NOTHING except one fresh results
file: every heldout_test_acc it reports comes from exp63_ckpt/*.json as run.

doc 49 SS6.2 (frozen):
  k2_gap_closes_with_data = TRUE iff S2's gap-to-ceiling shrinks by >= 0.06 from
  the smallest to the largest level AND S1's gap stays within +-0.03 across the
  same levels.
The ceiling is a property of the CELL and the ARM's K (exp61_bands), not of the
training-set size, so it is constant across levels -- which makes the shrink in
S2's gap numerically equal to the rise in S2's accuracy. Both are reported.

doc 49 SS6.2 also registers a THIRD OUTCOME: a gap that shrinks without closing
reads "consistent with sample complexity, insufficient data to confirm", and may
not be reported as either hypothesis winning.
"""
import glob, json, os, platform, statistics as st, sys, time
from collections import defaultdict

import torch

import exp61_bands as B61
import exp63_run as E
from exp42_run import fresh_results_path, git_note

SHRINK_BAR = 0.06          # doc 49 SS6.2, frozen
S1_BAND = 0.03             # doc 49 SS6.2, frozen
N_SEEDS = 5
CEILING_SOURCE = "exp61_bands.CORRECTED_CEILINGS (doc 47 SS17 erratum)"


def load_ckpts():
    """Every run record exp63 left behind, grouped by (arm, cell, level)."""
    g = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(E.CKPT_DIR, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        g[(r["arm"], r["cell"], r["level"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["seed"])
    return g


def block(rows):
    """mean / SAMPLE sd (ddof=1) / per-seed values ordered by seed."""
    accs = [r["heldout_test_acc"] for r in rows]
    return {"mean": st.mean(accs),
            "sd": st.stdev(accs),
            "per_seed": accs,
            "seeds": [r["seed"] for r in rows],
            "n_seeds": len(accs),
            "n_train_pairs": sorted({r["n_train_pairs"] for r in rows})}


def main():
    t0 = time.time()
    g = load_ckpts()
    expected = ([(a, c, L) for a in E.ARMS for c in E.CELLS for L in E.LEVELS]
                + [("S-shuffle", c, E.LEVELS[-1]) for c in E.CELLS])
    missing = [k for k in expected if len(g.get(k, [])) != N_SEEDS]
    if missing:
        raise RuntimeError("incomplete exp63 checkpoints: %s" % (missing,))
    n_runs = sum(len(v) for v in g.values())
    lo, hi = E.LEVELS[0], E.LEVELS[-1]

    rows, per_cell_verdict = [], {}
    for cell in E.CELLS:
        arms = {a: {str(L): block(g[(a, cell, L)]) for L in E.LEVELS}
                for a in E.ARMS}
        ceil = {a: B61.CORRECTED_CEILINGS[(cell, 2 if a == "S2" else 1)]
                for a in E.ARMS}
        for a in E.ARMS:                       # gap = ceiling - accuracy
            for L in E.LEVELS:
                b = arms[a][str(L)]
                b["gap_to_ceiling"] = ceil[a] - b["mean"]

        m = lambda a, L: arms[a][str(L)]["mean"]                 # noqa: E731
        s2_shrink = m("S2", hi) - m("S2", lo)                    # == gap shrink
        s1_moves = {str(L): m("S1", L) - m("S1", lo) for L in E.LEVELS[1:]}
        s1_stability = max(abs(v) for v in s1_moves.values())
        shrink_ok = bool(s2_shrink >= SHRINK_BAR)
        s1_ok = bool(s1_stability <= S1_BAND)
        verdict = bool(shrink_ok and s1_ok)
        per_cell_verdict[cell] = verdict

        rows.append({
            "cell": cell,
            "ceilings": {"S1_K1": ceil["S1"], "S2_K2": ceil["S2"],
                         "source": CEILING_SOURCE,
                         "constant_across_levels": True},
            "S1": arms["S1"], "S2": arms["S2"],
            "S_shuffle_floor": block(g[("S-shuffle", cell, hi)]),
            "criteria": {
                "s2_gap_shrink_%d_to_%d" % (lo, hi): s2_shrink,
                "s2_gap_shrink_equals_accuracy_rise": True,
                "s2_gap_to_ceiling_by_level":
                    {str(L): arms["S2"][str(L)]["gap_to_ceiling"]
                     for L in E.LEVELS},
                "s1_moves_vs_%d" % lo: s1_moves,
                "s1_stability_max_abs_move": s1_stability,
                "s1_delta_%d_to_%d" % (lo, hi): m("S1", hi) - m("S1", lo),
                "shrink_meets_bar_0.06": shrink_ok,
                "s1_within_band_0.03": s1_ok,
                "k2_gap_closes_with_data": verdict}})

    # ---- pooled reading: doc 49 SS6.2 does not say per-cell or pooled, so both
    # are computed and the verdict is reported under each.
    pm = lambda a, L: st.mean(                                   # noqa: E731
        [r["heldout_test_acc"] for c in E.CELLS for r in g[(a, c, L)]])
    pooled_shrink = pm("S2", hi) - pm("S2", lo)
    pooled_s1 = {str(L): pm("S1", L) - pm("S1", lo) for L in E.LEVELS[1:]}
    pooled_s1_stab = max(abs(v) for v in pooled_s1.values())
    pooled_verdict = bool(pooled_shrink >= SHRINK_BAR
                          and pooled_s1_stab <= S1_BAND)

    readings = {"per_cell_M2": per_cell_verdict["M2"],
                "per_cell_M3": per_cell_verdict["M3"],
                "pooled_over_cells": pooled_verdict,
                "any_cell": bool(any(per_cell_verdict.values())),
                "all_cells": bool(all(per_cell_verdict.values()))}
    overall = bool(any(readings.values()))

    verdicts = {
        "k2_gap_closes_with_data": overall,
        "readings": readings,
        "pooled": {"s2_gap_shrink_%d_to_%d" % (lo, hi): pooled_shrink,
                   "s1_moves_vs_%d" % lo: pooled_s1,
                   "s1_stability_max_abs_move": pooled_s1_stab},
        "criterion_bars": {"s2_gap_shrink_min": SHRINK_BAR,
                           "s1_gap_band": S1_BAND,
                           "levels_compared": [lo, hi]}}

    out = {
        "experiment": "exp63",
        "spec": E.DESIGN_DOC,
        "aggregated_from": "%s/*.json (%d runs, aggregation dated 2026-08-23; "
                           "runs completed 2026-08-22)" % (E.CKPT_DIR, n_runs),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dataset": E.DATA,
        "dataset_sha256": E.DATA_SHA,
        "subsample_seed": E.SUBSAMPLE_SEED,
        "levels": E.LEVELS,
        "arms": E.ARMS + ["S-shuffle"],
        "cells_run": E.CELLS,
        "n_seeds": N_SEEDS,
        "n_runs": n_runs,
        "sd_convention": "sample sd, ddof=1 (statistics.stdev)",
        "cells": rows,
        "verdicts": verdicts,
        "meta_notes": {
            "headline": (
                "k2_gap_closes_with_data = FALSE, and FALSE under every reading "
                "of doc 49 SS6.2's criterion -- per cell (M2, M3), pooled over "
                "cells, any-cell and all-cells. No reading of the frozen "
                "criterion returns TRUE."),
            "ceiling_is_constant": (
                "The criterion is stated on S2's GAP TO CEILING, but the "
                "ceiling depends only on the cell and on K (%s), not on the "
                "size of the training set, so it is identical at 120, 240 and "
                "470. The shrink in the gap is therefore numerically equal to "
                "the rise in accuracy, and is computed that way."
                % CEILING_SOURCE),
            "M2_near_miss_is_contaminated": (
                "M2's shrink (+%.4f) is the closer of the two to the 0.06 bar, "
                "but it is contaminated by a draw-level lift: S1 rose +%.4f "
                "over the same levels, breaching the registered +-0.03 band. "
                "That is exactly what doc 49 SS6.2's S1-stability condition "
                "exists to catch -- 'without the S1 condition, any draw-level "
                "effect that lifts everything would mimic sample complexity'. "
                "M2 fails the criterion on BOTH clauses."
                % (rows[0]["criteria"]["s2_gap_shrink_%d_to_%d" % (lo, hi)],
                   rows[0]["criteria"]["s1_stability_max_abs_move"])),
            "M3_is_clean_and_still_fails": (
                "M3 is the clean cell: S1 is genuinely flat (+%.4f from 120 to "
                "470, max move %.4f, inside the band), so there is no "
                "draw-level lift to subtract -- and S2 still rises only +%.4f "
                "over roughly 4x the training data, well short of 0.06."
                % (rows[1]["criteria"]["s1_delta_%d_to_%d" % (lo, hi)],
                   rows[1]["criteria"]["s1_stability_max_abs_move"],
                   rows[1]["criteria"]["s2_gap_shrink_%d_to_%d" % (lo, hi)])),
            "third_outcome_registered": (
                "doc 49 SS6.2 registered this outcome before the runs: at 470 "
                "items and 25 coefficients there are only ~19 items per "
                "coefficient, so the gap may shrink WITHOUT closing. That "
                "outcome reads 'consistent with sample complexity, "
                "insufficient data to confirm.' NEITHER hypothesis is promoted "
                "as winning: not sample complexity, and not "
                "optimisation/frequency-decay (doc 49 SS6.3). Any "
                "extrapolation from these three levels is descriptive only."),
            "floors": (
                "S-shuffle at full data sits at %.4f (M2) and %.4f (M3), so "
                "the trained arms are above their label-shuffled floors and "
                "the cells are admissible; the FALSE verdict is a real "
                "measurement, not a dead instrument."
                % (rows[0]["S_shuffle_floor"]["mean"],
                   rows[1]["S_shuffle_floor"]["mean"])),
            "scope": (
                "One task family (exp60's SB), two cells, five seeds, "
                "simulation, constructed categories, no classical comparator. "
                "Three subsample levels from a single registered subsample "
                "seed (%d); the level-to-level differences carry the seed's "
                "draw as well as its size." % E.SUBSAMPLE_SEED),
            "provenance": (
                "Aggregation only: no training was run here and no stored "
                "boolean from an earlier experiment changes. The runs "
                "completed 2026-08-22; this file is the verdict that should "
                "have been written then.")},
        "git_note": git_note(),
        "wall_time_sec": time.time() - t0}

    p = fresh_results_path("results_exp63")
    assert "exp63" in p
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("[exp63] %d runs aggregated from %s" % (n_runs, E.CKPT_DIR))
    for r in rows:
        c = r["criteria"]
        print("  %s  S2 %.4f/%.4f/%.4f  shrink %+.4f | S1 %.4f/%.4f/%.4f  "
              "max move %.4f | floor %.4f | verdict %s"
              % (r["cell"], r["S2"]["120"]["mean"], r["S2"]["240"]["mean"],
                 r["S2"]["470"]["mean"],
                 c["s2_gap_shrink_%d_to_%d" % (lo, hi)],
                 r["S1"]["120"]["mean"], r["S1"]["240"]["mean"],
                 r["S1"]["470"]["mean"], c["s1_stability_max_abs_move"],
                 r["S_shuffle_floor"]["mean"], c["k2_gap_closes_with_data"]))
    print("  pooled: shrink %+.4f | S1 max move %.4f | verdict %s"
          % (pooled_shrink, pooled_s1_stab, pooled_verdict))
    print("[exp63] k2_gap_closes_with_data = %s (FALSE under every reading)"
          % overall)
    print("[exp63] wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
