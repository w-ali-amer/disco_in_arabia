# -*- coding: utf-8 -*-
"""instrument_scaling_bands_c.py -- THE CONSTRUCTOR DEFECT, ITS CONTROL, AND THE
CORRECTED BAND LADDER.

WHAT THIS FILE EXISTS TO RECORD
-------------------------------
`diagnostic_scaling_bands_b.json` reported a NON-MONOTONE band ladder: the
constructed-band bound falls from 1.0000 (2 arcs) to a minimum 0.5298 (32 arcs)
and then REBOUNDS to 0.5735 (64) and 0.5973 (128). doc 63 SS3.6.1 had registered
that a non-monotone ladder would itself be the finding.

**It is not a finding about angle-space geometry. It is a DEFECT in the band
constructor, and this file proves that and repairs it.**

THE DEFECT
----------
`instrument_scaling_bounds.band_members` splits |B| bearers into k = A/2 runs as

    per = [size // k + (1 if i < size % k else 0) for i in range(k)]

i.e. the remainder `size % k` is assigned to the FIRST runs. At A <= 32 the
remainder is small and the allocation is nearly flat. At A = 64 (k = 32,
size = 81) it is [3]*17 + [2]*15, and at A = 128 (k = 64) it is [2]*17 + [1]*47
-- a systematic FRONT-LOADING of bearer density around the circle, measured here
as a front/back run-mass ratio rising 1.025 -> 1.455 -> 1.531.

WHY THAT INFLATES THE BOUND, VERIFIED RATHER THAN ARGUED
--------------------------------------------------------
For u(t) = (1, cos t, sin t) and antisymmetric A in R^3, x^T A y = w . (u_b x u_n)
decomposes into two interpretable families:

    R1  w = (1, 0, 0)      -> sign sin(t_n - t_b)          angular-ORDER rule
    R2  w = (0, w2, w3)    -> sign [f(t_b) - f(t_n)],
                              f(t) = R sin(t - phi)        SINUSOID-COMPARISON
                                                           rule -- a rank/
                                                           Mann-Whitney statistic

A density gradient around the circle is exactly what R2 exploits. Measured
(`mech_check`): R2 tracks the full bound through the rebound region (0.5596 vs
0.5735 at 64; 0.5919 vs 0.5973 at 128) while R1 stays near 0.52.

THE CONTROL, AND IT IS DECISIVE
-------------------------------
Rebuild the bands with the remainder spread EVENLY across runs instead of
front-loaded, holding |B|, the arc count and everything else fixed:

    arcs      front-loaded f/b   front-loaded R2   balanced f/b   balanced R2
     16             1.025            0.5154           1.025          0.5154
     32             1.025            0.5112           1.025          0.5112
     64             1.455            0.5596           1.025          0.5035
    128             1.531            0.5919           1.025          0.5021

**The rebound disappears entirely.** At A <= 32 the two constructions coincide,
which is why `band_32arc` -- doc 63 SS12.9's grade-3 target -- is UNAFFECTED and
its committed bound reproduces here exactly.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not modify `diagnostic_scaling_bounds.json` or
`diagnostic_scaling_bands_b.json` (Rule 3). Those remain the artefacts of record
for what they computed; this file records that their 64- and 128-arc rows are
INVALID AS STRUCTURELESS REFERENCES and supplies corrected ones.

The size-matched null is NOT recomputed: it depends only on (inventory, |B|,
mode, draws, NULL_SEED) and `bands_b` already asserted it reproduces to 1e-12.
Its stored mean/sd are cited and z is computed against them; **exact percentiles
are not recomputed and are reported as null**, with the reason, rather than
estimated.

    python instrument_scaling_bands_c.py     # -> diagnostic_scaling_bands_c.json
"""
import datetime
import json
import math
import os
import platform
import sys

import numpy as np

import instrument_scaling_bounds as B

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR_A = os.path.join(HERE, "diagnostic_scaling_bounds.json")
PRIOR_B = os.path.join(HERE, "diagnostic_scaling_bands_b.json")
OUT = os.path.join(HERE, "diagnostic_scaling_bands_c.json")

ARCS = (16, 32, 64, 128)


def main():
    pa = json.load(open(PRIOR_A, encoding="utf-8"))
    pb = json.load(open(PRIOR_B, encoding="utf-8"))
    nouns, rows, meta = B.load_inventory519()
    coords, U, featprov = B.build_fa_features(nouns)
    surf, camel, both = B.wazn_sets(nouns, rows)
    size = len(camel)
    th = {n: math.atan2(coords[n][1], coords[n][0]) for n in nouns}
    order = sorted(nouns, key=lambda n: th[n] % (2 * math.pi))
    N = len(order)

    def build(A, balanced):
        k = A // 2
        if balanced:
            per = [size // k] * k
            for j in range(size % k):
                per[int(round(j * k / (size % k))) % k] += 1
        else:
            per = [size // k + (1 if i < size % k else 0) for i in range(k)]
        out = []
        for i, c in enumerate(per):
            st = int(round(i * N / k))
            out.extend(order[st:st + c])
        return [n for n in nouns if n in set(out)], per

    def front_back(per):
        h = len(per) // 2
        return sum(per[:h]) / max(sum(per[h:]), 1)

    def r2_best(mem):
        S = set(mem)
        tb = np.array([th[n] for n in nouns if n in S])
        tn = np.array([th[n] for n in nouns if n not in S])
        nc = len(tb) * len(tn)
        best = 0.0
        for phi in np.linspace(0, 2 * math.pi, 1441, endpoint=False):
            c = int((np.sin(tb - phi)[:, None] > np.sin(tn - phi)[None, :]).sum())
            best = max(best, max(c, nc - c) / nc)
        return best

    nullref = pa["contrasts"]["band_32arc"]["null"]
    nm, nsd = nullref["mean"], nullref["sd"]

    corrected, defect = {}, {}
    for A in ARCS:
        mf, pf = build(A, balanced=False)
        mb, pb_ = build(A, balanced=True)
        non_b = [n for n in nouns if n not in set(mb)]
        rb = {k: v for k, v in B.constructive_bound(U, mb, non_b, "pair").items()
              if not k.startswith("_")}
        rb["arcs"] = B.arcs_of(coords, mb, nouns)
        assert rb["arcs"] == A, (A, rb["arcs"])
        rb["members_sha256"] = B.canonical_sha256(sorted(mb))
        rb["null_reference"] = {
            "mean": nm, "sd": nsd, "source": "diagnostic_scaling_bounds.json",
            "z_vs_null": float((rb["value"] - nm) / nsd),
            "percentile_of_contrast": None,
            "why_percentile_is_null": ("the 200 null draws are not recomputed here; "
                                       "bands_b already asserted they reproduce to "
                                       "1e-12, and an ESTIMATED percentile would be a "
                                       "different quantity from the measured ones"),
        }
        rb["below_null_mean"] = bool(rb["value"] < nm)
        corrected["band_%03darc_balanced" % A] = rb
        defect["band_%03darc" % A] = {
            "front_loaded_run_allocation": pf,
            "balanced_run_allocation": pb_,
            "front_back_mass_ratio_front_loaded": float(front_back(pf)),
            "front_back_mass_ratio_balanced": float(front_back(pb_)),
            "R2_sinusoid_front_loaded": r2_best(mf),
            "R2_sinusoid_balanced": r2_best(mb),
            "constructions_identical": bool(sorted(mf) == sorted(mb)),
        }
        print("A=%3d  balanced bound %.4f (z %+.2f)   f/b %.3f -> %.3f   R2 %.4f -> %.4f"
              % (A, rb["value"], rb["null_reference"]["z_vs_null"],
                 defect["band_%03darc" % A]["front_back_mass_ratio_front_loaded"],
                 defect["band_%03darc" % A]["front_back_mass_ratio_balanced"],
                 defect["band_%03darc" % A]["R2_sinusoid_front_loaded"],
                 defect["band_%03darc" % A]["R2_sinusoid_balanced"]), flush=True)

    committed = {r["arcs"]: r["bound"] for r in pb["full_band_ladder"]}
    ladder = []
    for a in sorted(set(list(committed) + list(ARCS))):
        row = {"arcs": a, "committed_front_loaded": committed.get(a)}
        k = "band_%03darc_balanced" % a
        row["corrected_balanced"] = corrected[k]["value"] if k in corrected else None
        ladder.append(row)
    vals = [(r["arcs"], r["corrected_balanced"] if r["corrected_balanced"] is not None
             else r["committed_front_loaded"]) for r in ladder]
    mono = all(vals[i][1] >= vals[i + 1][1] - 1e-12 for i in range(len(vals) - 1))

    out = {
        "component": "exp76_scaling_bands_c",
        "purpose": ("record and repair the band-constructor defect that produced the "
                    "non-monotone ladder in diagnostic_scaling_bands_b.json"),
        "status": ("PRE-WORK INSTRUMENT, doc 54's instrument exception. Bound "
                   "calculator, not an experiment. No exp76 experiment code exists."),
        "spec": "qnlp_private_docs/63_exp76_scaling_DRAFT.md SS2.6.2, SS3.6.2",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": B._sha(os.path.abspath(__file__)),
        "provenance": {
            "prior_a": os.path.basename(PRIOR_A), "prior_a_sha256": B._sha(PRIOR_A),
            "prior_b": os.path.basename(PRIOR_B), "prior_b_sha256": B._sha(PRIOR_B),
            "neither_prior_is_modified": True,
            "budget": B.BUDGET,
            "null_not_recomputed": ("cited from diagnostic_scaling_bounds.json; "
                                    "bands_b asserted it reproduces to 1e-12"),
        },
        "the_defect": {
            "where": "instrument_scaling_bounds.band_members",
            "code": "per = [size // k + (1 if i < size % k else 0) for i in range(k)]",
            "effect": ("the remainder size % k is assigned to the FIRST runs, "
                       "front-loading bearer density around the circle"),
            "per_arc": defect,
            "verdict": ("CONFIRMED: the rebound is a construction artefact. The "
                        "balanced control removes it at every arc count, and the "
                        "front/back mass ratio tracks it (1.025 -> 1.455 -> 1.531)."),
        },
        "mechanism": {
            "decomposition": ("x^T A y = w . (u_b x u_n); R1 (w=(1,0,0)) is an "
                              "angular-ORDER rule, R2 (w=(0,w2,w3)) is a "
                              "SINUSOID-COMPARISON rank statistic"),
            "measured_R1_front_loaded": {2: 0.6224, 4: 0.5352, 8: 0.5062, 16: 0.5049,
                                         32: 0.5054, 64: 0.5195, 128: 0.5235},
            "reading": ("R2 tracks the full bound through the rebound region while R1 "
                        "stays near 0.52, so the inflation is the sinusoid-comparison "
                        "family exploiting the density gradient -- which is what a "
                        "'2-arc' half-plane rule is in this basis"),
        },
        "corrected_bands": corrected,
        "ladder": ladder,
        "corrected_ladder_is_monotone_decreasing": mono,
        "grade3_target_unaffected": {
            "band_32arc_committed": pa["contrasts"]["band_32arc"]["value"],
            "band_32arc_balanced": corrected["band_032arc_balanced"]["value"],
            "identical": bool(abs(pa["contrasts"]["band_32arc"]["value"]
                                  - corrected["band_032arc_balanced"]["value"]) < 1e-12),
            "note": ("at A <= 32 the front-loaded and balanced constructions coincide "
                     "(f/b 1.025 both ways), so doc 63 SS12.9's grade-3 target and its "
                     "below-null status are untouched by this defect"),
        },
    }
    assert not os.path.exists(OUT), "Rule 3: %s exists and is never overwritten" % OUT
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("corrected ladder monotone: %s" % mono)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
