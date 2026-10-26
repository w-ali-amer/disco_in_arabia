# -*- coding: utf-8 -*-
"""readout_arity_bounds_addendum.py -- the exp68 feature maps, re-searched with
the stronger machinery.

WHY THIS FILE EXISTS
--------------------
`readout_arity_bounds.py` found that its M = 2 cell -- argmax over two outcomes,
which is `sign(<C, phi>)` with `C = V_0 - V_1` UNRESTRICTED -- is the SAME
optimisation problem as the committed PER-TWIN general-C bound, and that its
three-stage search returns strictly better values than the committed budgeted
search on the FROZEN-ANGLE (FA) map. Those supersessions are recorded in
`diagnostic_readout_arity_bounds.json`.

**But FA is only one of exp68's three registered feature maps.** exp68's arms
are F1 (PC1 only) and F2 (the two-qubit WIDTH register) on the RAW AraVec PCA
plane, and their per-twin bounds were never re-searched. F1's committed
nonoccupational bound -- 0.7625 = 122/160, a CONSTRUCTIVE lower bound, not an
exact one -- is the low end of the `F2 - F1` headroom exp68's design context
quotes, so if F1 moves, that headroom moves with it. This file closes that gap.

WHAT IS RECOMPUTED, AND WHAT IS NOT
-----------------------------------
Recomputed: the PER-TWIN bound (doc 47 SS17's population, denominator 2|B||N|)
for all four doc 47 SS13 categories on all three exp68 feature maps, using the
IDENTICAL feature formulas the committed instrument uses -- imported from
`instrument_sb_bounds`, never re-typed -- and the identical registered scaling
(`full`, theta = (p + 1) pi).

NOT recomputed, and NOT touched: `diagnostic_readout_arity_bounds.json` (a
finished artefact -- doc 47 rule 3: never overwrite a results file, a new
computation gets a new filename), `diagnostic_sb_bounds.json`,
`results_exp68.json`, or any exp68 script.

WHAT A MOVED BOUND DOES AND DOES NOT MEAN
-----------------------------------------
1. **No committed selftest breaks.** `capacity_check.ANCHORS` pins what THAT
   budgeted search deterministically returns, and that is unchanged. What moves
   is the BOUND -- the committed files now carry a weaker lower bound for the
   same quantity (doc 47 SS18.2: constructive bounds move with search effort;
   the stronger one supersedes).
2. **No exp68 verdict is recomputed here.** doc 54 SS12 declares these bounds
   CONTEXT ONLY and reads no criterion against them. This file states the
   corrected context and stops; re-reading any exp68 criterion is the
   orchestrator's call, not this instrument's.
3. **The pair-level EXACT ceilings are unaffected.** They are exact by
   arrangement enumeration in R^3, not by search.

CLI
---
    python readout_arity_bounds_addendum.py
"""
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import instrument_sb_bounds as isb                    # noqa: E402
import readout_arity_bounds as rab                    # noqa: E402

OUT = os.path.join(HERE, "diagnostic_readout_arity_bounds_addendum.json")

# Committed per-twin values, HAND-TYPED from the committed artefacts. Hand-typing
# is what makes this a reconciliation rather than a printout.
#   diagnostic_sb_bounds.json :: target_selection.table.<cat>.<map>.per_twin
COMMITTED = {
    "tertiary":        {"F1": (154, 190), "F2": (172, 190), "FA": (163, 190)},
    "uniformed":       {"F1": (124, 160), "F2": (148, 160), "FA": (140, 160)},
    "manual":          {"F1": (182, 190), "F2": (190, 190), "FA": (160, 190)},
    "nonoccupational": {"F1": (122, 160), "F2": (160, 160), "FA": (148, 160)},
}
# capacity_check.py :: ANCHORS -- the two per-twin anchors, hand-typed.
CAPACITY_CHECK_ANCHORS = {
    "exp68_F1_raw_per_twin": {"map": "F1", "category": "nonoccupational",
                              "expect": 0.7625, "fraction": "122/160"},
    "exp68_FA_per_twin": {"map": "FA", "category": "nonoccupational",
                          "expect": 0.9250, "fraction": "148/160"},
}


def per_twin_design(U, bearers, nonbearers):
    """Joint design over ALL ordered twins for an arbitrary per-referent map U.

    Row = vec(u_agent u_patient^T); label 1 = the 'yes' twin (agent = bearer).
    With M = 2 the argmax decision is sign(<V_0 - V_1, phi>) with V_0 - V_1
    unrestricted, which is EXACTLY doc 47 SS17's per-twin general-C system."""
    X, Y = [], []
    for b in bearers:
        for y in nonbearers:
            X.append(np.outer(U[b], U[y]).ravel()); Y.append(1)
            X.append(np.outer(U[y], U[b]).ravel()); Y.append(0)
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=int)


def agent_only_design_U(U, bearers, nonbearers):
    """The agent factor of the same design -- the small-space warm-start stage,
    identical in role to `readout_arity_bounds.agent_only_design`."""
    X, Y = [], []
    for b in bearers:
        for y in nonbearers:
            X.append(U[b]); Y.append(1)
            X.append(U[y]); Y.append(0)
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=int)


def main():
    t0 = time.time()
    seed = 51000
    BUD = {"anneal_restarts": 256, "anneal_steps": 1200,
           "hinge_restarts": 512, "hinge_steps": 1200,
           "hop_restarts": 256, "hop_steps": 960,
           "polish": 1200, "hops": 2,
           "margin_ladder": list(rab.MARGIN_LADDER)}

    coords_raw, prov_raw = isb.load_raw()
    scaling = isb.DEFAULT_SCALING
    rng = np.random.default_rng(seed + 7)

    maps = {}
    for kind in ("F1", "F2", "FA"):
        U, desc = isb.features(coords_raw, kind, scaling)
        maps[kind] = {"U": U, "definition": desc,
                      "d": len(next(iter(U.values())))}

    table = {}
    for key, cat in isb.CATEGORIES.items():
        B = cat["members"]
        N = [n for n in isb.HUMANS if n not in B]
        n_twins = 2 * len(B) * len(N)
        row = {"gloss": cat["gloss"], "size": len(B), "n_twins": n_twins}
        for kind in ("F1", "F2", "FA"):
            U = maps[kind]["U"]
            d = maps[kind]["d"]
            PHI, y = per_twin_design(U, B, N)
            PA, yA = agent_only_design_U(U, B, N)
            assert PHI.shape == (n_twins, d * d)
            grp = [0, 1]

            # exact pair-level (d = 3 only) -- the invariant the per-twin bound
            # must never fall below.
            pl = isb.bound(U, B, N, "pair")
            # the committed per-twin witness, handed in as a seed so this search
            # is guaranteed NOT to return anything worse than the committed value
            tw = isb.bound(U, B, N, "twin")
            Cw = tw["_c"].reshape(d, d)

            seeds = [np.stack([Cw.ravel(), -Cw.ravel()])]
            A = ((Cw - Cw.T) / 2.0).ravel()
            seeds.append(np.stack([A, -A]))
            seeds = rab.with_rotations(seeds)

            # stage 0: agent-only warm start in R^d, then lift
            sw_seeds = rab.with_rotations([]) + [rng.standard_normal((2, d))
                                                 for _ in range(24)]
            cw, Vw, _lg = rab.search_cell(PA, yA, 2, grp, sw_seeds, BUD,
                                          seed + 11 * d)
            lift = np.zeros((2, d * d))
            e0 = np.eye(d)[0]
            for m in range(2):
                lift[m] = np.outer(Vw[m], e0).ravel()
            seeds = seeds + rab.with_rotations([lift])

            cnt, V, slog = rab.search_cell(PHI, y, 2, grp, seeds, BUD,
                                           seed + 13 * d)
            acc, orient, _n = rab.hard_acc(V, PHI, y, grp)
            cc, cd = COMMITTED[key][kind]
            committed = cc / cd
            assert cd == n_twins, "denominator mismatch for %s/%s" % (key, kind)
            delta = acc - committed
            row[kind] = {
                "definition": maps[kind]["definition"],
                "d_per_referent": d, "d_joint": d * d,
                "value": acc, "fraction": "%d/%d" % (cnt, n_twins),
                "count": int(cnt), "denominator": n_twins,
                "orientation": orient,
                "committed_value": committed,
                "committed_fraction": "%d/%d" % (cc, cd),
                "delta": delta,
                "supersedes": bool(delta > 1e-12),
                "never_below_committed": bool(delta >= -1e-12),
                "at_or_above_pair_level": bool(acc >= pl["value"] - 1e-12),
                "pair_level_exact_if_available": (pl["value"] if pl["exact"] else None),
                "pair_level_fraction": pl["fraction"],
                "method": ("CONSTRUCTIVE LOWER BOUND (same three-stage search as "
                           "readout_arity_bounds.py: annealed logsumexp surrogate, "
                           "latent multiclass hinge over a 4-margin ladder, "
                           "hard-count polish, plus basin hops; SEEDED with the "
                           "committed per-twin witness, so it can never return "
                           "less than the committed value)"),
                "exact": False,
                "strict_margin_certification": rab.strict_margin_report(
                    V, PHI, y, grp, orient),
                "agent_only_warm_start": {"fraction": "%d/%d" % (cw, n_twins),
                                          "value": cw / n_twins},
                "stage_log": slog,
                "witness_V": V.tolist(),
            }
            print("[%-16s %s] %s = %.4f   committed %s = %.4f   %s"
                  % (key, kind, row[kind]["fraction"], acc,
                     row[kind]["committed_fraction"], committed,
                     "SUPERSEDES (+%.4f)" % delta if delta > 1e-12
                     else "unchanged"), flush=True)
        row["F2_minus_F1_recomputed"] = row["F2"]["value"] - row["F1"]["value"]
        row["F2_minus_F1_committed"] = (row["F2"]["committed_value"]
                                        - row["F1"]["committed_value"])
        row["F2_minus_FA_recomputed"] = row["F2"]["value"] - row["FA"]["value"]
        table[key] = row

    anchors = {}
    for name, spec in CAPACITY_CHECK_ANCHORS.items():
        cell = table[spec["category"]][spec["map"]]
        anchors[name] = {
            "committed_expect": spec["expect"],
            "committed_fraction": spec["fraction"],
            "recomputed": cell["value"], "recomputed_fraction": cell["fraction"],
            "anchor_still_passes": True,
            "why": ("capacity_check's anchors pin what THAT budgeted search "
                    "deterministically returns; that behaviour is unchanged and "
                    "the selftest still passes. What is superseded is the BOUND "
                    "the anchor's value represents."),
            "bound_superseded": cell["supersedes"],
        }

    best_sep_recomputed = max(table, key=lambda k: table[k]["F2_minus_F1_recomputed"])

    payload = {
        "instrument": "readout_arity_bounds_addendum.py",
        "script_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "companion": {
            "file": "diagnostic_readout_arity_bounds.json",
            "note": ("that file is FINISHED and was NOT modified. doc 47's rule 3 "
                     "(never overwrite a results file; a new computation gets a "
                     "new filename) applies to instrument outputs too."),
            "readout_arity_bounds_sha256": rab.__dict__.get("_sha_placeholder") or
            hashlib.sha256(open(os.path.join(HERE, "readout_arity_bounds.py"),
                                "rb").read()).hexdigest(),
        },
        "generated_for": ("closing the ONE supersession readout_arity_bounds.py "
                          "left open: it re-searched the FROZEN-ANGLE map only, "
                          "while exp68's arms are F1 and F2 on the RAW AraVec PCA "
                          "plane."),
        "not_a_results_file": ("named diagnostic_*, not results_*: an instrument "
                              "output, not an experiment result. No exp68 "
                              "criterion is read against it (doc 54 SS12 declares "
                              "these bounds CONTEXT ONLY)."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_sec": time.time() - t0,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "provenance": {
            "family": prov_raw,
            "registered_scaling": {"name": scaling,
                                   "formula": isb.SCALINGS[scaling][0]},
            "feature_formulas": {k: maps[k]["definition"] for k in maps},
            "feature_formulas_source": ("imported from instrument_sb_bounds.features, "
                                        "never re-typed here"),
            "instrument_sb_bounds_sha256": hashlib.sha256(
                open(os.path.join(HERE, "instrument_sb_bounds.py"), "rb").read()
            ).hexdigest(),
            "membership_closure": rab.MEMBERSHIP_CLOSURE,
        },
        "bound_definition": {
            "quantity": ("PER-TWIN reachable accuracy, doc 47 SS17: the two twins "
                         "of a cast are SEPARATE constraints on a GENERAL C, so a "
                         "cast may score 0, 1/2 or 1. Denominator = 2|B||N|."),
            "identity_used": ("argmax over M = 2 outcomes is sign(<V_0 - V_1, phi>) "
                              "with V_0 - V_1 unrestricted, so the M = 2 cell of the "
                              "readout-arity calculus IS the per-twin general-C "
                              "MAX-FS. That identity is why the stronger search "
                              "applies to a quantity two committed files carry."),
            "seeding": ("the committed per-twin witness is handed to the search as "
                        "a seed, so the returned value is >= the committed value BY "
                        "CONSTRUCTION and a 'supersedes' verdict can never be a "
                        "search artefact in the wrong direction"),
        },
        "search_budget": {"budget": BUD, "seed_base": seed,
                          "identical_across_categories_and_maps": True},
        "table": table,
        "capacity_check_anchor_impact": anchors,
        "headline": {
            "F1_nonoccupational_committed": COMMITTED["nonoccupational"]["F1"][0]
            / COMMITTED["nonoccupational"]["F1"][1],
            "F1_nonoccupational_recomputed": table["nonoccupational"]["F1"]["value"],
            "F1_nonoccupational_fraction": table["nonoccupational"]["F1"]["fraction"],
            "F2_minus_F1_nonoccupational_committed":
                table["nonoccupational"]["F2_minus_F1_committed"],
            "F2_minus_F1_nonoccupational_recomputed":
                table["nonoccupational"]["F2_minus_F1_recomputed"],
            "largest_F2_minus_F1_committed": "nonoccupational",
            "largest_F2_minus_F1_recomputed": best_sep_recomputed,
            "target_selection_rule_unchanged": bool(
                best_sep_recomputed == "nonoccupational"),
        },
        "any_superseded": bool(any(table[k][m]["supersedes"]
                                   for k in table for m in ("F1", "F2", "FA"))),
        "none_below_committed": bool(all(table[k][m]["never_below_committed"]
                                         for k in table for m in ("F1", "F2", "FA"))),
        "scope": ("Representational bounds over the 24 frozen human nouns, one task "
                  "family (SB), simulation, training-free. doc 54 SS12 reads NO exp68 "
                  "criterion against these numbers, and neither does this file: it "
                  "corrects the CONTEXT exp68's design quotes, nothing else."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote %s (%.1fs)" % (OUT, time.time() - t0))
    print("any bound superseded: %s | none below committed: %s"
          % (payload["any_superseded"], payload["none_below_committed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
