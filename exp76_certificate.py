# -*- coding: utf-8 -*-
"""exp76_certificate.py -- doc 63 SS9. PHASE 1'S REGISTERED ARTEFACT, AND WHAT
GATES PHASE B.

This is SS0.1's deliverable, and it is the reason phase 1 is worth running even
if every accuracy criterion comes back FALSE. Every plan to go beyond one qubit
per referent runs into the same unanswered question: IS THE STATE THIS CIRCUIT
PRODUCES REPRESENTABLE BY A MATRIX-PRODUCT STATE AT A TRACTABLE BOND DIMENSION?
The field's answer is a measurement, not an argument, and phase 1 makes it.

WHAT IS ALREADY MEASURED (SS9.2, by `exp76_run.forward_traced`, stored in every
per-run checkpoint): for each trained cell, each of the 5 seeds, each of the 300
held-out items, each of four stages -- intro, after event 1, after the last
event, at readout -- and each of the n - 1 contiguous wire-cuts, the Schmidt
spectrum, S(k) in bits, R_1e-6(k) and S_max_page(k). THIS FILE ONLY AGGREGATES.

WHAT THE CERTIFICATE REGISTERS: THE GROWTH RATE, NOT THE ABSOLUTE VALUE (SS9.3.1)

    S_max(n)  = max over seeds, items, stages, cuts of S(k), at cast size n
                (TRAINED checkpoints; the init value is measured separately)
    S_page(n) = floor(n / 2)              bits -- the maximal possible value
    f(n)      = S_max(n) / S_page(n)      the entanglement FILLING FRACTION

    least-squares fit over the five n values 2, 3, 4, 5, 7:  S_max(n) = a*n + b

    REGISTERED BOOLEAN, threshold named before the numbers existed:
          entanglement_area_law_like = ( a <= 0.10 bits per wire )

    REPORTED AS CONTEXT, no criterion reads it:
          chi_hat(W) = 2 ^ ( a*W + b )   at W = 16, 24, 48

PROVENANCE OF THE 0.10 BAR (SS9.3.1), stated against the alternative it must
reject and computed over THE ACTUAL GRID POINTS rather than a textbook n/2: a
volume-law state saturates S_page(n) = floor(n/2), and least-squares over the
grid's own five values -- (2,1), (3,1), (4,2), (5,2), (7,3) -- gives a MAXIMAL
ACHIEVABLE SLOPE of 0.4189 bits per wire, not 0.5. 0.10 is 24% of that.

SS9.3.2 THE FIT IS ACROSS n_active AND THE T CONFOUND IS NAMED. The grid is
triangular, so larger n_active occurs only at larger T. The fit is reported BOTH
pooled and stratified by T, THE BOOLEAN READS THE POOLED FIT, and if the two
stratified slopes disagree by more than 0.05 the boolean is REPORTED AS
DISAGREEING rather than resolved silently.

SS9.3 `mps_feasible_at_chi64` IS STRUCTURALLY INERT AND IS REGISTERED AS SUCH: a
contiguous cut of an n-wire state has Schmidt rank at most 2^min(k+1, n-k-1), so
across the registered grid the rank can never exceed 8 and chi = 64 is TRUE BY
ARITHMETIC. It is stored for completeness. NO READING DEPENDS ON IT.

SS9.7 WHAT SS9 IS NOT: `entanglement_area_law_like` is a pre-registered boolean
stored like every other, but it is NOT a criterion about the experiment's
outcome. No accuracy boolean reads it, no accuracy boolean feeds it, and neither
may be promoted onto the other (the exp44 no-promotion clause). It is an
INFRASTRUCTURE DECISION INSTRUMENT.

Writes `diagnostic_exp76_certificate.json`. IT WRITES NO RESULTS FILE.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np

import exp76_run as R
import exp76_datagen as D76

OUT = "diagnostic_exp76_certificate.json"
SLOPE_BAR = R.AREA_LAW_SLOPE_BAR                    # 0.10 bits per wire
STRATUM_DISAGREEMENT_BAR = 0.05                     # SS9.3.2
CHI_WIDTHS = (16, 24, 48)


def lsq_slope(ns, ys):
    """S_max(n) = a*n + b, ordinary least squares."""
    x = np.asarray(ns, float)
    y = np.asarray(ys, float)
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.abs(A @ np.array([a, b]) - y).max())
    return float(a), float(b), resid


def load_runs(ckpt_dir, seeds):
    runs = {}
    for cell in R.TRAINED_CELLS:
        per = []
        for s in range(seeds):
            f = os.path.join(ckpt_dir, "%s_s%d.json" % (cell, s))
            if not os.path.exists(f):
                raise RuntimeError(
                    "the certificate is incomplete: %s is missing. doc 63 "
                    "SS20.4: aggregation runs only when all 120 checkpoints "
                    "exist." % f)
            per.append(json.load(open(f, encoding="utf-8")))
        runs[cell] = per
    return runs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    runs = load_runs(a.ckpt_dir, a.seeds)
    rep = {"component": "exp76 entanglement certificate (doc 63 SS9)",
           "spec": R.SPEC, "status": "PHASE 1 DELIVERABLE",
           "platform": platform.platform(), "python": platform.python_version(),
           "numpy": np.__version__,
           "eps": R.SCHMIDT_EPS, "stages": list(R.CERT_STAGES),
           "per_cell": {}}

    # ---------------------------------------------------- per cell, per stage
    for cell, per in runs.items():
        cfg = D76.CELL_CFG[cell]
        smax = max(r["certificate_doc63_SS9"]["S_max_overall"] for r in per)
        rmax = max(r["certificate_doc63_SS9"]["R_max_overall"] for r in per)
        arg = max((r["certificate_doc63_SS9"]["argmax"] for r in per),
                  key=lambda d: d["S"])
        init = [r.get("certificate_at_init_doc63_SS9_4") for r in per]
        init = [x for x in init if x]
        stages = {}
        for st in R.CERT_STAGES:
            vals = [r["certificate_doc63_SS9"]["stages"][st] for r in per
                    if st in r["certificate_doc63_SS9"]["stages"]]
            if vals:
                stages[st] = {
                    "S_max": float(max(v["S_max"] for v in vals)),
                    "S_mean": float(np.mean([v["S_mean"] for v in vals])),
                    "S_p95": float(np.mean([v["S_p95"] for v in vals])),
                    "R_max": int(max(v["R_max"] for v in vals))}
        page = cfg["n"] // 2
        rep["per_cell"][cell] = {
            "n": cfg["n"], "T": cfg["T"], "axis": cfg["axis"],
            "inventory": cfg["inv"], "category": cfg["cat"],
            "n_seeds": len(per),
            "S_max": smax, "R_max": rmax, "S_page": page,
            "filling_fraction": (smax / page) if page else None,
            "argmax": arg, "stages": stages,
            "S_max_at_init": (max(x["S_max_overall"] for x in init)
                              if init else None),
            "mps_feasible_at_chi64": bool(rmax <= R.CHI_INERT)}

    # -------------------------------------------- SS9.3.1's fit, over the grid
    grid = [c for c in R.TRAINED_CELLS if D76.CELL_CFG[c]["axis"] == "grid"]
    by_n = {}
    for cell in grid:
        n = D76.CELL_CFG[cell]["n"]
        by_n.setdefault(n, []).append(rep["per_cell"][cell]["S_max"])
    ns = sorted(by_n)
    smax_n = [max(by_n[n]) for n in ns]
    a_pool, b_pool, res_pool = lsq_slope(ns, smax_n)

    strat = {}
    for T in (4, 6):
        cells = [c for c in grid if D76.CELL_CFG[c]["T"] == T]
        xs = sorted(D76.CELL_CFG[c]["n"] for c in cells)
        ys = [rep["per_cell"]["n%d_T%d" % (n, T)]["S_max"] for n in xs]
        aa, bb, rr = lsq_slope(xs, ys)
        strat["T%d" % T] = {"n_values": xs, "S_max": ys,
                            "slope_a": aa, "intercept_b": bb, "max_resid": rr}
    disagree = abs(strat["T4"]["slope_a"] - strat["T6"]["slope_a"])

    # the maximal achievable slope over THESE grid points, recomputed not typed
    a_page, b_page, _r = lsq_slope(ns, [n // 2 for n in ns])

    init_pairs = {c: (rep["per_cell"][c]["S_max_at_init"],
                      rep["per_cell"][c]["S_max"]) for c in grid}
    trained_exceeds_init = all(
        (v[0] is None) or (v[1] >= v[0] - 1e-9) for v in init_pairs.values())

    rep["growth_doc63_SS9_3_1"] = {
        "n_values": ns, "S_max_by_n": smax_n,
        "S_page_by_n": [n // 2 for n in ns],
        "filling_fraction_by_n": [smax_n[i] / (ns[i] // 2) if ns[i] // 2 else None
                                  for i in range(len(ns))],
        "pooled": {"slope_a": a_pool, "intercept_b": b_pool,
                   "max_resid": res_pool},
        "stratified_by_T": strat,
        "stratified_slope_disagreement": disagree,
        "stratified_disagreement_bar": STRATUM_DISAGREEMENT_BAR,
        "strata_disagree": bool(disagree > STRATUM_DISAGREEMENT_BAR),
        "maximal_achievable_slope_over_this_grid": a_page,
        "bar_as_fraction_of_maximal": SLOPE_BAR / a_page if a_page else None,
        "chi_hat_context_only": {
            "W%d" % W: float(2.0 ** (a_pool * W + b_pool)) for W in CHI_WIDTHS},
        "init_versus_trained": {
            "per_cell": {c: {"S_max_at_init": v[0], "S_max_trained": v[1]}
                         for c, v in init_pairs.items()},
            "trained_at_least_init_everywhere": bool(trained_exceeds_init),
            "reading": ("doc 63 SS9.4 row 4: S_max small at init and large after "
                        "training means TRAINING IS WHAT MAKES THE STATE HARD, "
                        "and any MPS plan must budget for the trained regime, "
                        "never the initial one.")},
    }

    # ------------------------------------------------- SS12.10's two booleans
    area_law = bool(a_pool <= SLOPE_BAR)
    rep["verdicts"] = {
        "entanglement_area_law_like": area_law,
        "entanglement_area_law_like_reason": (
            "pooled least-squares slope a = %.4f bits per wire over n_active in "
            "%r, against the registered bar a <= %.2f (doc 63 SS9.3.1, named "
            "before the numbers existed). %s"
            % (a_pool, ns, SLOPE_BAR,
               ("The two stratified slopes disagree by %.4f > %.2f, so THE "
                "BOOLEAN IS REPORTED AS DISAGREEING rather than resolved "
                "silently (SS9.3.2): width and depth cannot be separated by "
                "this grid and the certificate returns a RANGE, with phase B's "
                "chi set from the LARGER slope."
                % (disagree, STRATUM_DISAGREEMENT_BAR))
               if disagree > STRATUM_DISAGREEMENT_BAR else
               ("The two stratified slopes agree to %.4f, inside the %.2f bar."
                % (disagree, STRATUM_DISAGREEMENT_BAR)))),
        "mps_feasible_at_chi64": all(v["mps_feasible_at_chi64"]
                                     for v in rep["per_cell"].values()),
        "mps_feasible_at_chi64_is_INERT": (
            "doc 63 SS9.3: TRUE BY ARITHMETIC at every phase-1 cell (8 <= 64 "
            "always). Computed and stored for completeness and REGISTERED AS "
            "INERT -- the fourth use of doc 58 SS9.5's precedent, and the FIRST "
            "time this programme has caught an inert boolean at DESIGN TIME "
            "instead of after the compute. NO READING DEPENDS ON IT."),
    }

    rep["disposition_doc63_SS9_4"] = {
        "observed": ("a = %.4f, strata %s"
                     % (a_pool, "disagree" if disagree >
                        STRATUM_DISAGREEMENT_BAR else "agree")),
        "row": ("a <= 0.10, both strata agreeing -- THE COMPOSITION BLOCKS "
                "PRODUCE SUB-VOLUME ENTANGLEMENT and it does not fill as width "
                "grows; phase B's infrastructure is justified by a measurement, "
                "with chi_hat(24) as its design target"
                if area_law and disagree <= STRATUM_DISAGREEMENT_BAR else
                "the two stratified slopes disagree -- WIDTH AND DEPTH CANNOT BE "
                "SEPARATED BY THIS GRID; the certificate returns a range rather "
                "than a number and phase B's chi is set from the LARGER slope"
                if disagree > STRATUM_DISAGREEMENT_BAR else
                "a approaching 0.5, f(n) -> 1 -- VOLUME LAW: the states fill "
                "their entanglement capacity and MPS buys nothing at any width. "
                "A FINDING, and it retires the phase-B plan on evidence rather "
                "than on effort; doc 59 SS2's MPS assignment is amended IN "
                "WRITING")}

    rep["scoping_doc63_SS9_6"] = [
        "CIRCUIT-LEVEL entropy of the evolving state. arXiv:2307.06937 "
        "distinguishes it from the model's coefficient-MPS entropy and they "
        "differ; SS9.2 measures the STATE and no function-level claim is made.",
        "THE INFERENCE RUNS ONE WAY ONLY. arXiv:2607.08396 exhibits "
        "high-entropy circuits that are nonetheless efficiently MPS-simulable, "
        "so a low R_1e-6 soundly certifies FEASIBILITY -- the only direction "
        "SS9.3 uses -- and a high one certifies nothing about hardness.",
        "TRAINABILITY AND MPS-FEASIBILITY ARE NOT INDEPENDENT CLAIMS. "
        "arXiv:2312.09121 (Nature Communications 16, 7907, 2025) argues that "
        "provable absence of barren plateaus points toward classical "
        "simulability, so `gap_stable_in_n` TRUE together with "
        "`entanglement_area_law_like` TRUE is ONE FINDING WEARING TWO NAMES, "
        "not two corroborating results, and NO WRITE-UP MAY PRESENT THEM AS "
        "MUTUAL SUPPORT. It costs this programme nothing it had: SS16 forbids "
        "any advantage reading regardless.",
        "SS23 item 5: the certificate measures the states THESE circuits "
        "produce and nothing more. If phase B changes the encoding -- which is "
        "the entire point of phase B -- the certificate does not transfer, so "
        "it justifies phase B's INFRASTRUCTURE only for the encodings phase 1 "
        "ran.",
    ]
    rep["not_a_criterion_doc63_SS9_7"] = (
        "no accuracy boolean reads this, no accuracy boolean feeds it, and "
        "neither may be promoted onto the other (the exp44 no-promotion "
        "clause). SS12.10: read LAST, gating nothing in SS18.4's chain.")

    json.dump(rep, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp76 cert] a = %.4f bits/wire (bar %.2f) -> "
          "entanglement_area_law_like = %s" % (a_pool, SLOPE_BAR, area_law))
    print("[exp76 cert] stratified T4 %.4f / T6 %.4f (disagreement %.4f)"
          % (strat["T4"]["slope_a"], strat["T6"]["slope_a"], disagree))
    print("[exp76 cert] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
