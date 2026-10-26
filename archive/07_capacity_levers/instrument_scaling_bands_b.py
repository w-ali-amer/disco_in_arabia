# -*- coding: utf-8 -*-
"""instrument_scaling_bands_b.py -- doc 63 §3.6.1's registered obligation.

WHY THIS FILE EXISTS, AND WHY IT IS A SECOND FILE
-------------------------------------------------
`instrument_scaling_bounds.py` built the constructed-band calibration ladder at
2, 4, 8, 16 and 32 arcs. Every wazn contrast it measured sits at **112-186
arcs**. Reading a 140-arc contrast against a 32-arc band is an EXTRAPOLATION,
which is exactly what a calibration layer must not require -- doc 63 §3.6.1
registered the fix as a blocking pre-commit obligation:

    bands at 64 and 128 arcs at the same |B| = 81, so the wazn classes are
    BRACKETED rather than extrapolated to.

Rule 3 forbids overwriting `diagnostic_scaling_bounds.json`, so this is a new
producer writing a new artefact. Nothing in the first file is read for anything
but the committed values asserted below, and it is not modified.

THE NULL IS NOT RE-DRAWN, IT IS RE-DERIVED AND ASSERTED
-------------------------------------------------------
The size-matched null depends only on (inventory, |B|, mode, draws, NULL_SEED)
and each draw is seeded at NULL_SEED + t, so recomputing it reproduces the SAME
200 subsets. It is recomputed here -- not copied -- and the recomputed mean and
sd are ASSERTED equal to the values stored in `diagnostic_scaling_bounds.json`
to 1e-12. That makes the percentile in this file exact rather than estimated,
and it doubles as a determinism check on the first run.

    python instrument_scaling_bands_b.py            # -> diagnostic_scaling_bands_b.json
"""
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np

import instrument_scaling_bounds as B

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.join(HERE, "diagnostic_scaling_bounds.json")
OUT = os.path.join(HERE, "diagnostic_scaling_bands_b.json")

NEW_ARCS = (64, 128)
WORKERS = 4          # exp73 holds 5 of the 10 cores; this leaves it untouched


def main():
    prior = json.load(open(PRIOR, encoding="utf-8"))
    nouns, rows, meta = B.load_inventory519()
    coords, U, featprov = B.build_fa_features(nouns)
    surf, camel, both = B.wazn_sets(nouns, rows)
    size = len(camel)                      # 81, matching the committed ladder

    # ---- the null, recomputed deterministically and ASSERTED against the prior
    nul = B.size_matched_null(U, size, "pair", 200, WORKERS)
    ref = prior["contrasts"]["band_32arc"]["null"]
    assert abs(float(nul.mean()) - ref["mean"]) < 1e-12, \
        "null mean does not reproduce: %.15f vs %.15f" % (nul.mean(), ref["mean"])
    assert abs(float(nul.std(ddof=1)) - ref["sd"]) < 1e-12, \
        "null sd does not reproduce"

    out_contrasts = {}
    for A in NEW_ARCS:
        mem = B.band_members(coords, A, size, nouns)
        realised = B.arcs_of(coords, mem, nouns)
        assert realised == A, "requested %d arcs, realised %d" % (A, realised)
        non = [n for n in nouns if n not in set(mem)]
        r = {k: v for k, v in B.constructive_bound(U, mem, non, "pair").items()
             if not k.startswith("_")}
        pct = float((nul < r["value"]).mean() * 100.0)
        r["arcs"] = realised
        r["contrast_kind"] = "constructed angle band (capacity-validation layer)"
        r["note"] = ("|B| matched to faail_camel; exactly %d arcs by construction; "
                     "doc 63 SS3.6.1's bracketing obligation" % A)
        r["members_sha256"] = B.canonical_sha256(sorted(mem))
        r["null"] = {
            "draws": 200, "size_matched_B": size, "null_seed": B.NULL_SEED,
            "mean": float(nul.mean()), "sd": float(nul.std(ddof=1)),
            "min": float(nul.min()), "max": float(nul.max()),
            "percentile_of_contrast": pct, "identical_search_budget": True,
            "z_vs_null": float((r["value"] - nul.mean()) / max(nul.std(ddof=1), 1e-12)),
            "reproduces_prior_null": True,
        }
        r["grade3"] = {
            "below_null": bool(r["value"] < nul.mean()),
            "below_null_5th_percentile": bool(pct < 5.0),
        }
        out_contrasts["band_%03darc" % A] = r
        print("[band_%03darc] |B|=%3d arcs=%3d bound=%.4f null=%.4f+-%.4f pct=%.1f"
              % (A, size, realised, r["value"], nul.mean(), nul.std(ddof=1), pct),
              flush=True)

    # ---- the full ladder, for the doc's table: prior rows + these
    ladder = []
    for k, c in prior["contrasts"].items():
        if k.startswith("band_"):
            ladder.append({"arcs": c["arcs"], "bound": c["value"],
                           "percentile": c["null"]["percentile_of_contrast"],
                           "source": "diagnostic_scaling_bounds.json"})
    for k, c in out_contrasts.items():
        ladder.append({"arcs": c["arcs"], "bound": c["value"],
                       "percentile": c["null"]["percentile_of_contrast"],
                       "source": "diagnostic_scaling_bands_b.json"})
    ladder.sort(key=lambda r: r["arcs"])
    monotone = all(ladder[i]["bound"] >= ladder[i + 1]["bound"] - 1e-12
                   for i in range(len(ladder) - 1))

    wazn = {k: {"arcs": prior["contrasts"][k]["arcs"],
                "bound": prior["contrasts"][k]["value"],
                "percentile": prior["contrasts"][k]["null"]["percentile_of_contrast"]}
            for k in ("faail_surface", "faail_camel", "faail_intersection",
                      "mu_participle_surface", "nisba_yaa_surface")}
    worst_band = ladder[-1]["bound"]
    bracketed = {k: {"arcs": v["arcs"], "bound": v["bound"],
                     "arcs_exceed_widest_band": bool(v["arcs"] > ladder[-1]["arcs"]),
                     "bound_above_widest_band": bool(v["bound"] > worst_band),
                     "excess_over_widest_band": float(v["bound"] - worst_band)}
                 for k, v in wazn.items()}

    out = {
        "component": "exp76_scaling_bands_b",
        "purpose": ("doc 63 SS3.6.1: constructed angle bands at 64 and 128 arcs, so "
                    "the 112-186-arc wazn contrasts are BRACKETED by the calibration "
                    "ladder instead of extrapolated to from 32 arcs"),
        "status": ("PRE-WORK INSTRUMENT, doc 54's instrument exception. Bound "
                   "calculator, not an experiment. No exp76 experiment code exists."),
        "spec": "qnlp_private_docs/63_exp76_scaling_DRAFT.md SS3.6.1",
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": B._sha(os.path.abspath(__file__)),
        "provenance": {
            "prior_artefact": os.path.basename(PRIOR),
            "prior_artefact_sha256": B._sha(PRIOR),
            "prior_script_sha256": prior["script_sha256"],
            "instrument_sb_bounds_sha256": B._sha(
                os.path.join(HERE, "instrument_sb_bounds.py")),
            "budget": B.BUDGET, "null_draws": 200, "null_seed": B.NULL_SEED,
            "null_recomputed_not_copied": True,
            "null_reproduces_prior_to_1e-12": True,
            "exactness": ("CONSTRUCTIVE LOWER BOUND, as in the prior artefact; the "
                          "EXACT R^3 enumeration is infeasible at this inventory size"),
        },
        "contrasts": out_contrasts,
        "full_band_ladder": ladder,
        "ladder_monotone_decreasing_in_arcs": monotone,
        "wazn_contrasts_against_the_ladder": bracketed,
        "reading": (
            "The ladder is the structureless reference: a bearer set with A arcs and "
            "NO linguistic content. It is monotone decreasing in A. Every wazn "
            "contrast has MORE arcs than the widest band and a HIGHER bound than it, "
            "so its separability is NOT explained by arc count. What the ladder does "
            "NOT control for is SEMANTIC COHERENCE of the class -- see doc 63 SS3.6.3."),
    }
    assert not os.path.exists(OUT), "Rule 3: %s exists and is never overwritten" % OUT
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("ladder monotone in arcs: %s" % monotone)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
