# -*- coding: utf-8 -*-
"""capacity_validation.py -- prediction vs measurement, over every committed
(computed bound, trained held-out accuracy) pair the programme has produced.

WHAT THIS IS
------------
The capacity instrument (doc 47 SS3/SS17, `instrument_sb_bounds.py`,
`exp61_bands.py`, `capacity_check.py`) predicts a REACHABLE ACCURACY before any
training. Seven experiments have since trained circuits whose held-out accuracy
can be read against such a bound. This script assembles all of those pairs into
one table and one figure, so the instrument can be validated as an instrument.

ANALYSIS OF FROZEN ARTIFACTS ONLY. Nothing is trained here and no existing file
is modified. Every number in the output JSON is RE-READ from its committed
source file at runtime; the hand-typed constants in this script are ASSERTIONS
against those re-read values, which is what makes this a cross-check rather than
a transcription.

WHAT A ROW MEANS, AND WHAT IT DOES NOT
--------------------------------------
A row pairs ONE bound with ONE trained mean. `conversion_ratio = measured/bound`
answers "how much of the representationally available accuracy did training
actually convert?". It is NOT a claim that the bound is tight, and it is NOT a
claim that the two quantities are over the same population unless
`bound_population` says so. doc 47 SS18.5 is binding:

    A BOUND IS STATED TOGETHER WITH THE POPULATION IT BOUNDS.

Three bound populations appear here and they are NOT interchangeable:
  * BAND CENTRES        exp60, exp61, exp63 -- LP-exact over the band geometry.
                        doc 47 SS18.1: this is NOT a bound over a finite sampled
                        noun inventory, and exp61's CONFIRM stage measured
                        M2w/S1 = 0.8880 ABOVE the 0.8750 band ceiling for
                        exactly that reason. (That confirm stage lives in doc 47
                        SS18 and in no committed results_*.json, so it is not a
                        row here; it is recorded in `caveats`.)
  * HELD-OUT NOUN-PAIR ANGLES   exp62 -- constructive LOWER bounds over the
                        actual angles of that draw. Different draws give
                        different bounds; each row carries its own dataset tag.
  * PER-TWIN MAX-FS OVER THE REFERENT FEATURE MAP   exp68 -- doc 54 SS12,
                        recomputed live here by `capacity_check`.
  * DOC-REGISTERED      exp65 (doc 51 SS11.1), exp67 (doc 53 SS2). Computed in a
                        session whose code was not committed; carried with that
                        provenance stated, never as if re-derivable here.

LEVER FAMILIES -- the assignment rule, stated once
--------------------------------------------------
A row's family is decided by WHAT SUPPLY THE LEVER ADDS, not by the arm's name:

  K1-single-wire      the referent enters through exactly ONE RY carrying ONE
                      angle: d = 3, spectrum K = 1.
  K2-within-wire      extra SPECTRUM for the same one angle -- re-uploading, or
                      the same theta applied twice. K = 2. (exp62's W-wide is
                      here, not under width: doc 48 SS5 registers it as
                      `RY(theta)` on BOTH qubits of the register with the SAME
                      theta, which buys spectrum, not information.)
  width-register      extra WIRES at one RY per wire -- workspace (exp62 W-work,
                      K = 1) or genuinely independent per-qubit angles (exp68
                      F2, theta0 = PC1 and theta1 = PC2).
  M-ary-readout       exp67's readout-arity lever (R2 / R4 / R8).
  classical-comparator  a classical model fitted on the SAME features.
  floor               label-shuffled / severed controls. Carried with the bound
                      of the arm they are matched to; a floor is not expected to
                      convert anything, and its presence in the table is what
                      shows the y-axis is a real measurement.

CLI
---
    python capacity_validation.py                 # JSON + figure
    python capacity_validation.py --no-figure
    python capacity_validation.py --strict        # any failed assertion exits 1
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import capacity_check                        # noqa: E402
import exp61_bands                           # noqa: E402
import instrument_sb_bounds as isb           # noqa: E402

OUT_JSON = os.path.join(HERE, "diagnostic_capacity_validation.json")
FIG_DATA = os.path.join(HERE, "fig_capacity_validation_data.json")
FIG_SVG = os.path.join(HERE, "fig_capacity_validation.svg")
FIG_PNG = os.path.join(HERE, "fig_capacity_validation.png")

OVER_BOUND_MARGIN = 0.06        # exp61 SS8 / exp67 SS4 / doc 54 SS7's shared bar

FAMILY_RULE = {
    "K1-single-wire": "one RY, one angle, d = 3, spectrum K = 1",
    "K2-within-wire": "extra spectrum for the SAME angle (re-upload, or the same "
                      "theta on two qubits) -- K = 2, no new information",
    "width-register": "extra wires at ONE RY per wire (workspace, or independent "
                      "per-qubit angles)",
    "M-ary-readout": "doc 53's readout-arity lever",
    "classical-comparator": "a classical model on the SAME features",
    "floor": "label-shuffled / severed control, matched to an arm's bound",
}


# --------------------------------------------------------------------- utils
class Recorder(object):
    """Reads every value out of a file and checks it against a hand-typed
    expectation. The expectation is the check; the file is the truth."""

    def __init__(self):
        self.checks = []
        self.failures = []
        self.files = {}

    def load(self, name):
        path = os.path.join(HERE, name)
        with open(path, "rb") as fh:
            raw = fh.read()
        self.files[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw)}
        return json.loads(raw.decode("utf-8"))

    def check(self, label, got, want, tol=5e-4):
        ok = (got is not None and abs(float(got) - float(want)) <= tol)
        self.checks.append({"label": label, "expected": want, "read": got,
                            "tol": tol, "passed": bool(ok)})
        if not ok:
            self.failures.append(label)
        return ok


def _n_ckpt_seeds(pattern):
    """n_seeds counted from the committed per-seed checkpoint records.

    A silent 0 here would look like a legitimate 'unknown', so it is an error:
    the caller asked for a count that the artifacts are supposed to supply.
    """
    n = len(glob.glob(os.path.join(HERE, pattern)))
    if n == 0:
        raise AssertionError("no checkpoints matched %r -- n_seeds cannot be "
                             "counted, and a silent 0 must not be published"
                             % pattern)
    return n


SUBFAMILY = {
    # reviewer-ratified 2026-08-24: "width-register" pools two different things and
    # the JSON outlives the conversation. W-work's second qubit carries NOTHING
    # (spectator); exp68 F2 is the programme's ONLY two-feature register, so any
    # "parallel wires convert" claim about INFORMATION rests on one experiment.
    "spectator-qubit": "second wire present but carries no feature (exp62 W-work)",
    "two-feature-register": "each wire carries a DIFFERENT feature (exp68 F2 only)",
}


def width_subfamily(experiment, arm):
    if arm == "W-work":
        return "spectator-qubit"
    if experiment == "exp68" and arm == "F2":
        return "two-feature-register"
    return None


def row(experiment, arm, cell, family, K, bound, bound_prov, measured, sd,
        n_seeds, source_file, measured_field, dataset_tag, role="primary",
        note=None):
    r = {
        "experiment": experiment, "arm": arm, "cell": cell,
        "family": family, "family_rationale": FAMILY_RULE[family],
        "subfamily": width_subfamily(experiment, arm), "K": K,
        "bound": None if bound is None else float(bound),
        "bound_provenance": bound_prov,
        "measured_mean": None if measured is None else float(measured),
        "measured_sd": None if sd is None else float(sd),
        "n_seeds": n_seeds,
        "source_results_file": source_file,
        "measured_field": measured_field,
        "dataset_tag": dataset_tag,
        "role": role,
    }
    if bound and measured is not None:
        r["conversion_ratio"] = float(measured) / float(bound)
        r["gap"] = float(measured) - float(bound)
        r["over_bound"] = bool(r["gap"] > 0)
        r["over_bound_flag"] = bool(r["gap"] > OVER_BOUND_MARGIN)
    else:
        r["conversion_ratio"] = r["gap"] = None
        r["over_bound"] = r["over_bound_flag"] = False
    if note:
        r["note"] = note
    return r


def prov(file, field, kind, population, note=None):
    p = {"file": file, "field": field, "kind": kind, "population": population}
    if note:
        p["note"] = note
    return p


# ------------------------------------------------------------------- exp60
def rows_exp60(R):
    d = R.load("results_exp60.json")
    tag = "stories_exp60_SB(+confirm) / %s" % d["confirm_dataset_sha256"][:8]
    out = []
    # exp60's angle map is ONE bearer band and ONE non-bearer band with enforced
    # disjointness (doc 46 SS6), i.e. m = 1 -- the M1 cell of exp61's geometry,
    # whose corrected per-twin ceiling is LP-exact and equal to 1.0 at K = 1 and
    # K = 2 alike. The bound is therefore COMPUTED, not asserted.
    b1 = exp61_bands.CORRECTED_CEILINGS[("M1", 1)]
    b2 = exp61_bands.CORRECTED_CEILINGS[("M1", 2)]
    R.check("exp60 bound M1/K1 = 1.0", b1, 1.0)
    R.check("exp60 bound M1/K2 = 1.0", b2, 1.0)
    bp = lambda K: prov(                                            # noqa: E731
        "exp61_bands.py", "CORRECTED_CEILINGS[('M1', %d)]" % K, "exact",
        "band centres, m = 1 per class",
        "exp60's constructed map is one disjoint band per class (doc 46 SS6, "
        "band disjointness ASSERTED by the generator), so its geometry is "
        "exp61's M1 and the ceiling is 1.0 for every K. doc 46 SS6 records the "
        "same value as the generator's per-cell bayes_ceiling.")
    for stage, seeds_key in (("confirm", "confirm_seeds"), ("screen", "screen_seeds")):
        for c in d[stage]:
            for arm, K, fam in (("S-base", 1, "K1-single-wire"),
                                ("S-reup", 2, "K2-within-wire")):
                a = c[arm]
                out.append(row(
                    "exp60", arm, "%s [%s]" % (c["cell"], stage), fam, K,
                    b1 if K == 1 else b2, bp(K), a["heldout_mean"],
                    a["heldout_sd"], a["n_seeds"], "results_exp60.json",
                    "%s[cell=%s].%s.heldout_mean" % (stage, c["cell"], arm), tag,
                    role="primary" if stage == "confirm" else "context"))
            out.append(row(
                "exp60", "S-shuffle floor", "%s [%s]" % (c["cell"], stage),
                "floor", 1, b1, bp(1), c["floor_mean"], c["floor_sd"],
                d["config"][seeds_key], "results_exp60.json",
                "%s[cell=%s].floor_mean" % (stage, c["cell"]), tag,
                role="floor"))
    # the two registered must-fail controls
    for arm in ("S-flat", "S-hash"):
        a = d["controls"][arm]
        out.append(row("exp60", arm, "D0.40_w0.05 [screen]", "floor", 1, b1,
                       bp(1), a["heldout_mean"], a["heldout_sd"],
                       len(a["per_seed"]), "results_exp60.json",
                       "controls.%s.heldout_mean" % arm, tag, role="floor",
                       note="registered must-fail control, not the S-shuffle floor"))
    conf = {c["cell"]: c for c in d["confirm"]}
    R.check("exp60 confirm D0.40_w0.05 S-base",
            conf["D0.40_w0.05"]["S-base"]["heldout_mean"], 1.0)
    R.check("exp60 confirm D0.40_w0.05 S-reup",
            conf["D0.40_w0.05"]["S-reup"]["heldout_mean"], 0.998667)
    return out


# ------------------------------------------------------------------- exp61
def rows_exp61(R):
    d = R.load("results_exp61.json")
    tag = "stories_exp60_SB.json / %s" % d["dataset_sha256"][:8]
    out = []
    for c in d["cells"]:
        cell = c["cell"]
        for arm, K, fam in (("S1", 1, "K1-single-wire"),
                            ("S2", 2, "K2-within-wire")):
            a = c[arm]
            corrected = exp61_bands.CORRECTED_CEILINGS[(cell, K)]
            superseded = d["frozen_ceilings"]["%s_K%d" % (cell, K)]
            R.check("exp61 superseded ceiling %s/K%d in-file" % (cell, K),
                    superseded, exp61_bands.FROZEN_CEILINGS[(cell, K)])
            R.check("exp61 in-file arm ceiling is the SUPERSEDED one %s/%s"
                    % (cell, arm), a["ceiling"], superseded)
            p = prov("exp61_bands.py", "CORRECTED_CEILINGS[('%s', %d)]" % (cell, K),
                     "exact", "band centres (doc 47 SS18.1: NOT a finite noun "
                     "inventory)",
                     "BOUND OF RECORD (doc 47 SS17 erratum). results_exp61.json's "
                     "frozen_ceilings.%s_K%d = %.4f is the SUPERSEDED pair-level "
                     "LOWER bound and is not used here." % (cell, K, superseded))
            p["superseded_value_in_results_file"] = superseded
            out.append(row("exp61", arm, cell, fam, K, corrected, p,
                           a["heldout_mean"], a["heldout_sd"], a["n_seeds"],
                           "results_exp61.json",
                           "cells[cell=%s].%s.heldout_mean" % (cell, arm), tag))
        out.append(row("exp61", "S-shuffle floor", cell, "floor", 1,
                       exp61_bands.CORRECTED_CEILINGS[(cell, 1)],
                       prov("exp61_bands.py",
                            "CORRECTED_CEILINGS[('%s', 1)]" % cell, "exact",
                            "band centres"),
                       c["floor_mean"], c["floor_sd"], d["config"]["seeds"],
                       "results_exp61.json",
                       "cells[cell=%s].floor_mean" % cell, tag, role="floor"))
    cells = {c["cell"]: c for c in d["cells"]}
    R.check("exp61 M2/S1 held-out", cells["M2"]["S1"]["heldout_mean"], 0.682667)
    R.check("exp61 M2w/S2 held-out", cells["M2w"]["S2"]["heldout_mean"], 0.850667)
    R.check("exp61 M3/S1 held-out", cells["M3"]["S1"]["heldout_mean"], 0.480667)
    R.check("exp61 corrected M2/K1", exp61_bands.CORRECTED_CEILINGS[("M2", 1)], 0.875)
    R.check("exp61 corrected M3/K1", exp61_bands.CORRECTED_CEILINGS[("M3", 1)], 7 / 9)
    return out


# ------------------------------------------------------------------- exp62
def rows_exp62(R):
    out = []

    # ---- C (original draw). Bounds are the NOUN-LEVEL constructive lower
    #      bounds stored IN this file, not band ceilings.
    d = R.load("results_exp62_c.json")
    tag = "exp62-C draw / %s" % d["dataset_sha256"][:8]
    for c in d["cells"]:
        cell = c["cell"]
        bK = {1: c["noun_bound_K1"], 2: c["noun_bound_K2"]}
        bp = lambda K: prov(                                        # noqa: E731
            "results_exp62_c.json", "cells[cell=%s].noun_bound_K%d" % (cell, K),
            "constructive",
            "held-out NOUN-PAIR angles of the exp62-C draw (doc 47 SS18.1/SS18.2)",
            "a constructive LOWER bound from a budgeted search, computed on THIS "
            "draw's angles; it is not the exp61 band ceiling and it does not "
            "transfer to another draw")
        for arm, K, fam in (("W-work", 1, "width-register"),
                            ("W-wide", 2, "K2-within-wire")):
            a = c[arm]
            R.check("exp62-C %s/%s in-file bound == cell noun_bound_K%d"
                    % (cell, arm, K), a["bound"], bK[K])
            out.append(row("exp62-C", arm, cell, fam, K, a["bound"], bp(K),
                           a["mean"], a["sd"],
                           _n_ckpt_seeds("exp62c_ckpt/%s_%s_s*.json" % (arm, cell)),
                           "results_exp62_c.json",
                           "cells[cell=%s].%s.mean" % (cell, arm), tag,
                           note="n_seeds counted from exp62c_ckpt/%s_%s_s*.json"
                                % (arm, cell)))
        for arm, K, fam in (("S1", 1, "K1-single-wire"),
                            ("S2", 2, "K2-within-wire")):
            out.append(row("exp62-C", arm, cell, fam, K, bK[K], bp(K),
                           c["%s_mean" % arm], None, None,
                           "results_exp62_c.json",
                           "cells[cell=%s].%s_mean" % (cell, arm), tag,
                           role="context",
                           note="only the MEAN is stored in this file (no sd, no "
                                "n_seeds); carried as context, not as a primary row"))

    # ---- D (fresh-draw replication). Its bounds are ITS OWN.
    d = R.load("results_exp62_d.json")
    tag = "exp62-D fresh draw / %s" % d["dataset_sha256"][:8]
    for c in d["cells"]:
        cell = c["cell"]
        bK = {1: c["bound_K1"], 2: c["bound_K2"]}
        bp = lambda K: prov(                                        # noqa: E731
            "results_exp62_d.json", "cells[cell=%s].bound_K%d" % (cell, K),
            "constructive",
            "held-out NOUN-PAIR angles of the exp62-D FRESH draw",
            "fresh-draw bound: a different sample of held-out nouns, so this is "
            "not the same number as exp62-C's even where the value coincides")
        for arm, K, fam in (("S1", 1, "K1-single-wire"),
                            ("W-work", 1, "width-register"),
                            ("S2", 2, "K2-within-wire"),
                            ("W-wide", 2, "K2-within-wire")):
            a = c[arm]
            R.check("exp62-D %s/%s in-file bound == cell bound_K%d"
                    % (cell, arm, K), a["bound"], bK[K])
            out.append(row("exp62-D", arm, cell, fam, K, a["bound"], bp(K),
                           a["mean"], a["sd"],
                           _n_ckpt_seeds("exp62d_ckpt/%s_%s_s*.json" % (arm, cell)),
                           "results_exp62_d.json",
                           "cells[cell=%s].%s.mean" % (cell, arm), tag,
                           note="n_seeds counted from exp62d_ckpt/%s_%s_s*.json"
                                % (arm, cell)))
        out.append(row("exp62-D", "S-shuffle floor", cell, "floor", 1,
                       bK[1], bp(1), c["floor_mean"], c["floor_sd"],
                       _n_ckpt_seeds("exp62d_ckpt/S-shuffle_%s_s*.json" % cell),
                       "results_exp62_d.json",
                       "cells[cell=%s].floor_mean" % cell, tag, role="floor"))

    cc = {c["cell"]: c for c in R.load("results_exp62_c.json")["cells"]}
    dd = {c["cell"]: c for c in R.load("results_exp62_d.json")["cells"]}
    R.check("exp62-C M2 W-work mean", cc["M2"]["W-work"]["mean"], 0.776042)
    R.check("exp62-C M2 W-work bound", cc["M2"]["W-work"]["bound"], 0.75)
    R.check("exp62-C M3 noun_bound_K1", cc["M3"]["noun_bound_K1"], 0.6875)
    R.check("exp62-D M2 S2 mean", dd["M2"]["S2"]["mean"], 0.722917)
    R.check("exp62-D M3 W-wide mean", dd["M3"]["W-wide"]["mean"], 0.704167)
    return out


# ------------------------------------------------------------------- exp63
def rows_exp63(R):
    d = R.load("results_exp63.json")
    tag = "%s / %s (subsampled levels)" % (d["dataset"], d["dataset_sha256"][:8])
    full = max(d["levels"])
    out = []
    for c in d["cells"]:
        cell = c["cell"]
        src = c["ceilings"]["source"]
        R.check("exp63 %s S1 ceiling == exp61_bands corrected" % cell,
                c["ceilings"]["S1_K1"], exp61_bands.CORRECTED_CEILINGS[(cell, 1)])
        R.check("exp63 %s S2 ceiling == exp61_bands corrected" % cell,
                c["ceilings"]["S2_K2"], exp61_bands.CORRECTED_CEILINGS[(cell, 2)])
        for arm, K, fam in (("S1", 1, "K1-single-wire"),
                            ("S2", 2, "K2-within-wire")):
            b = c["ceilings"]["%s_K%d" % (arm, K)]
            p = prov("results_exp63.json",
                     "cells[cell=%s].ceilings.%s_K%d" % (cell, arm, K), "exact",
                     "band centres (doc 47 SS18.1)",
                     "the file itself records the source as %r" % src)
            for lvl in sorted(c[arm], key=int):
                a = c[arm][lvl]
                out.append(row("exp63", arm, "%s @ %s items" % (cell, lvl), fam,
                               K, b, p, a["mean"], a["sd"], a["n_seeds"],
                               "results_exp63.json",
                               "cells[cell=%s].%s.%s.mean" % (cell, arm, lvl),
                               tag,
                               role="primary" if int(lvl) == full else "context"))
        f = c["S_shuffle_floor"]
        out.append(row("exp63", "S-shuffle floor", "%s @ %d items" % (cell, full),
                       "floor", 1, c["ceilings"]["S1_K1"],
                       prov("results_exp63.json",
                            "cells[cell=%s].ceilings.S1_K1" % cell, "exact",
                            "band centres"),
                       f["mean"], f["sd"], f["n_seeds"], "results_exp63.json",
                       "cells[cell=%s].S_shuffle_floor.mean" % cell, tag,
                       role="floor"))
    cells = {c["cell"]: c for c in d["cells"]}
    R.check("exp63 M2/S1 @470", cells["M2"]["S1"]["470"]["mean"], 0.759333)
    R.check("exp63 M3/S2 @470", cells["M3"]["S2"]["470"]["mean"], 0.632667)
    return out


# ------------------------------------------------------------------- exp65
def rows_exp65(R):
    d = R.load("results_exp65.json")
    tag = "stories_exp65_WZ.json / %s" % d["dataset_sha256"][:8]
    b = 1.0
    p = prov("qnlp_private_docs/51_exp65_wazn_binding_PROPOSAL.md",
             "SS11.1 table, row 'positional', column 'faa3il bound'",
             "doc-registered",
             "the 8 held-out words of exp65's WORD-holdout split",
             "computed in the doc 51 SS11 session; that code was not committed, "
             "so it is carried as DOC-REGISTERED and is not re-derived here. "
             "doc 51 SS11.3 records that 15.8% of random 3-vs-5 splits also "
             "reach 1.0000 at n = 8, so this bound cannot test a CAP -- exp65's "
             "registered scope is transfer only.")
    out = [
        row("exp65", "S-real", "faa3il vs non-faa3il", "K1-single-wire", 1, b, p,
            d["arms"]["S-real"]["mean"], d["arms"]["S-real"]["sd"],
            len(d["arms"]["S-real"]["per_seed"]), "results_exp65.json",
            "arms.S-real.mean", tag),
        row("exp65", "char-positional logreg", "faa3il vs non-faa3il",
            "classical-comparator", None, b, p,
            d["classical_comparator"]["char_positional_logreg_heldout"], None, 1,
            "results_exp65.json",
            "classical_comparator.char_positional_logreg_heldout", tag,
            note="a single deterministic fit; no sd is stored"),
        row("exp65", "S-shuffle floor", "faa3il vs non-faa3il", "floor", 1, b, p,
            d["arms"]["S-shuffle"]["mean"], d["arms"]["S-shuffle"]["sd"],
            len(d["arms"]["S-shuffle"]["per_seed"]), "results_exp65.json",
            "arms.S-shuffle.mean", tag, role="floor"),
    ]
    R.check("exp65 S-real", d["arms"]["S-real"]["mean"], 1.0)
    R.check("exp65 classical comparator",
            d["classical_comparator"]["char_positional_logreg_heldout"], 1.0)
    return out


# ------------------------------------------------------------------- exp67
def rows_exp67(R):
    dq = R.load("results_exp67.json")
    dc = R.load("results_exp67_comparator.json")
    tag = "%s / %s" % (dc["dataset"], dq["dataset_sha256"][:8])
    R.check("exp67 and its comparator share a dataset sha (1.0 == same)",
            1.0 if dq["dataset_sha256"] == dc["dataset_sha256"] else 0.0, 1.0)
    ARITY = {"R2": 2, "R4": 4, "R8": 8}
    ccells = {c["cell"]: c for c in dc["cells"]}
    out = []
    for c in dq["cells"]:
        cell = c["cell"]
        cc = ccells[cell]
        for arm, M in ARITY.items():
            # the doc 53 SS2 registered ceiling, READ from the committed
            # comparator file's own record of it -- never hand-typed into output
            b = cc["comparator_M%d" % M]["doc53_SS2_registered_ceiling"]
            p = prov("results_exp67_comparator.json",
                     "cells[cell=%s].comparator_M%d.doc53_SS2_registered_ceiling"
                     % (cell, M),
                     "doc-registered (session-computed, empirically corroborated "
                     "by comparator)",
                     "band centres at readout arity M = %d (doc 53 SS2)" % M,
                     "doc 53 SS2's Davenport-Schinzel decision-complexity table. "
                     "The session code was not committed; the classical "
                     "comparator independently lands on it (doc 53 comparator "
                     "meta_notes.instrument_corroboration), which is the "
                     "corroboration, not a re-derivation.")
            a = c[arm]
            out.append(row("exp67", arm, cell, "M-ary-readout", 1, b, p,
                           a["mean"], a["sd"], len(a["per_seed"]),
                           "results_exp67.json",
                           "cells[cell=%s].%s.mean" % (cell, arm), tag))
            m = cc["comparator_M%d" % M]
            out.append(row("exp67", "comparator M=%d" % M, cell,
                           "classical-comparator", None, b, p, m["mean"],
                           m["sd"], len(m["per_seed"]),
                           "results_exp67_comparator.json",
                           "cells[cell=%s].comparator_M%d.mean" % (cell, M), tag,
                           note=m["status"]))
        out.append(row("exp67", "S-shuffle floor", cell, "floor", 1,
                       cc["comparator_M2"]["doc53_SS2_registered_ceiling"],
                       prov("results_exp67_comparator.json",
                            "cells[cell=%s].comparator_M2."
                            "doc53_SS2_registered_ceiling" % cell,
                            "doc-registered", "band centres, binary readout"),
                       c["S-shuffle"]["mean"], c["S-shuffle"]["sd"],
                       len(c["S-shuffle"]["per_seed"]), "results_exp67.json",
                       "cells[cell=%s].S-shuffle.mean" % cell, tag, role="floor"))
    R.check("exp67 M2 binary ceiling",
            ccells["M2"]["comparator_M2"]["doc53_SS2_registered_ceiling"], 0.8750)
    R.check("exp67 M3 binary ceiling",
            ccells["M3"]["comparator_M2"]["doc53_SS2_registered_ceiling"], 0.7778)
    R.check("exp67 M2 M=4 ceiling",
            ccells["M2"]["comparator_M4"]["doc53_SS2_registered_ceiling"], 1.0000)
    R.check("exp67 M3 M=4 ceiling",
            ccells["M3"]["comparator_M4"]["doc53_SS2_registered_ceiling"], 0.8400)
    R.check("exp67 M2 M=8 ceiling",
            ccells["M2"]["comparator_M8"]["doc53_SS2_registered_ceiling"], 1.0000)
    R.check("exp67 M3 M=8 ceiling",
            ccells["M3"]["comparator_M8"]["doc53_SS2_registered_ceiling"], 1.0000)
    qcells = {c["cell"]: c for c in dq["cells"]}
    R.check("exp67 M2/R8 measured", qcells["M2"]["R8"]["mean"], 0.738667)
    R.check("exp67 M3/R8 measured", qcells["M3"]["R8"]["mean"], 0.697333)
    R.check("exp67 M3 comparator M=8", ccells["M3"]["comparator_M8"]["mean"], 0.84)
    return out


# ------------------------------------------------------------------- exp68
def rows_exp68(R):
    d = R.load("results_exp68.json")
    diag = R.load("diagnostic_sb_bounds.json")
    tag = "%s / %s" % (d["dataset"], d["dataset_sha256"][:8])
    bounds = d["bounds_context_only_doc54_SS12"]["per_twin_bounds"]
    dtab = diag["target_selection"]["table"][d["target_category"]["key"]]

    # CROSS-CHECK 1: the results file and the instrument's own diagnostic agree.
    for k in ("F1", "F2", "FA"):
        R.check("exp68 bound %s: results file == diagnostic_sb_bounds" % k,
                bounds[k], dtab[k]["per_twin"], tol=5e-5)
    # CROSS-CHECK 2: capacity_check RECOMPUTES them live from the raw vectors.
    live = {}
    B = isb.CATEGORIES[d["target_category"]["key"]]["members"]
    for k in ("F1", "F2", "FA"):
        angles, _p = capacity_check._raw_angles(k)
        live[k] = capacity_check.capacity(angles, bearers=B, mode="twin",
                                          **capacity_check.DEFAULT_BUDGET)
        R.check("exp68 bound %s recomputed live by capacity_check" % k,
                live[k]["bound"], bounds[k], tol=5e-5)

    # SUPERSESSION (doc 47 SS18.2): the stronger three-stage search in
    # diagnostic_readout_arity_bounds_addendum.json lifted the FA per-twin
    # bound (M=2 argmax IS the general-C sign readout, same quantity).
    # F1 and F2 were re-searched there and did NOT move. The committed
    # 0.9250 checks below still pass (they pin capacity_check's budgeted
    # search); the BOUND OF RECORD for FA is the stronger value.
    add = R.load("diagnostic_readout_arity_bounds_addendum.json")
    sup = add["table"][d["target_category"]["key"]]
    bounds_of_record = dict(bounds)
    assert sup["FA"]["supersedes"] is True
    fa_new = sup["FA"]["value"]
    assert fa_new >= bounds["FA"] - 1e-12
    bounds_of_record["FA"] = fa_new
    R.check("exp68 FA bound SUPERSEDED value present in addendum",
            fa_new, 0.9625)

    FAM = {"F1": ("K1-single-wire", 1), "FA": ("K1-single-wire", 1),
           "F2": ("width-register", 1)}
    out = []
    for k in ("F1", "F2", "FA"):
        fam, K = FAM[k]
        p = prov("results_exp68.json",
                 "bounds_context_only_doc54_SS12.per_twin_bounds.%s" % k,
                 "exact" if live[k]["exact"] else "constructive",
                 "TWINS over the 4-bearer / 20-non-bearer human inventory "
                 "(doc 47 SS17 per-twin MAX-FS, doc 54 SS12)",
                 "recomputed live by capacity_check.capacity(mode='twin') to "
                 "%s (%s); also equals diagnostic_sb_bounds.json "
                 "target_selection.table.%s.%s.per_twin. doc 54 SS7: NO exp68 "
                 "criterion was read against this bound."
                 % (live[k]["fraction"], live[k]["method"],
                    d["target_category"]["key"], k))
        p["recomputed_here"] = live[k]["bound"]
        p["recomputed_fraction"] = live[k]["fraction"]
        if bounds_of_record[k] != bounds[k]:
            p["superseded_from"] = bounds[k]
            p["superseded_by"] = ("diagnostic_readout_arity_bounds_addendum.json"
                                  " :: supersessions.raw_family.%s.%s"
                                  % (d["target_category"]["key"], k))
            p["supersession_rule"] = ("doc 47 SS18.2: a constructive lower bound "
                                      "moves with search effort; the stronger "
                                      "value supersedes and is said, not found")
        a = d["arms"][k]
        out.append(row("exp68", k, d["target_category"]["key"], fam, K,
                       bounds_of_record[k], p, a["mean"], a["sd"], a["n_seeds"],
                       "results_exp68.json", "arms.%s.mean" % k, tag,
                       note="d = %d, %d wires, %d params"
                            % (live[k]["d"], a["wires"], a["n_params"])))
        m = d["comparator"]["arms"][k]
        out.append(row("exp68", "comparator %s" % k, d["target_category"]["key"],
                       "classical-comparator", None, bounds_of_record[k], p, m["mean"],
                       m["sd"], len(m["per_seed"]), "results_exp68.json",
                       "comparator.arms.%s.mean" % k, tag,
                       note="linear model on the IDENTICAL %d-dim features"
                            % m["feature_dim"]))
        f = d["floors"][k]
        out.append(row("exp68", "S-shuffle-%s floor" % k,
                       d["target_category"]["key"], "floor", K, bounds_of_record[k], p,
                       f["mean"], f["sd"], f["n_seeds"], "results_exp68.json",
                       "floors.%s.mean" % k, tag, role="floor",
                       note="MATCHED floor: same architecture, labels severed"))
    R.check("exp68 F1 bound", bounds["F1"], 0.7625)
    R.check("exp68 F2 bound", bounds["F2"], 1.0)
    R.check("exp68 FA bound", bounds["FA"], 0.9250)
    R.check("exp68 F1 measured", d["arms"]["F1"]["mean"], 0.656667)
    R.check("exp68 F2 measured", d["arms"]["F2"]["mean"], 0.958667)
    R.check("exp68 FA measured", d["arms"]["FA"]["mean"], 0.918667)
    R.check("exp68 comparator F2", d["comparator"]["arms"]["F2"]["mean"], 1.0)
    return out


# ------------------------------------------------------------------ assembly
def assemble():
    R = Recorder()
    rows = []
    for fn in (rows_exp60, rows_exp61, rows_exp62, rows_exp63, rows_exp65,
               rows_exp67, rows_exp68):
        rows.extend(fn(R))
    return rows, R


def summarise(rows, primary_only=False):
    fams = {}
    for r in rows:
        f = fams.setdefault(r["family"], {"n_rows": 0, "n_primary": 0,
                                          "ratios": [], "gaps": []})
        f["n_rows"] += 1
        if r["role"] == "primary":
            f["n_primary"] += 1
        if r["conversion_ratio"] is not None and r["role"] != "floor" \
                and (r["role"] == "primary" or not primary_only):
            f["ratios"].append(r["conversion_ratio"])
            f["gaps"].append(r["gap"])
    out = {}
    for k, v in sorted(fams.items()):
        rs = sorted(v["ratios"])
        out[k] = {"n_rows": v["n_rows"], "n_primary": v["n_primary"],
                  "n_with_ratio": len(rs),
                  "conversion_ratio_min": rs[0] if rs else None,
                  "conversion_ratio_median": rs[len(rs) // 2] if rs else None,
                  "conversion_ratio_mean": (sum(rs) / len(rs)) if rs else None,
                  "conversion_ratio_max": rs[-1] if rs else None,
                  "mean_gap": (sum(v["gaps"]) / len(v["gaps"])) if v["gaps"] else None}
    return out


def _by_population(rows):
    """The split that actually carries the reading (doc 47 SS18.5): a bound over
    band centres is not a bound over the inventory the circuit trained on, and
    conversion ratios against the two are not comparable."""
    out = {}
    for pop_key, want_band in (("band_centres", True),
                               ("trained_population", False)):
        for grp, fams in (("K1_and_width", ("K1-single-wire", "width-register")),
                          ("K2_and_Mary", ("K2-within-wire", "M-ary-readout")),
                          ("classical_comparator", ("classical-comparator",))):
            v = sorted(r["conversion_ratio"] for r in rows
                       if r["role"] == "primary" and r["family"] in fams
                       and r["conversion_ratio"] is not None
                       and r["bound_provenance"]["population"]
                       .startswith("band centres") == want_band)
            if v:
                out["%s / %s" % (pop_key, grp)] = {
                    "n": len(v), "min": v[0], "median": v[len(v) // 2],
                    "mean": sum(v) / len(v), "max": v[-1]}
    return out


CAVEATS = [
    "doc 47 SS18.5 (binding): a bound is stated together with the population it "
    "bounds. Rows whose bound_provenance.population is 'band centres' bound the "
    "BAND GEOMETRY, not the finite noun inventory the circuit was actually "
    "trained on. A measurement above such a bound is not necessarily an error.",
    "exp61's CONFIRM stage (doc 47 SS18, fresh draw sha 99fdafb3) measured "
    "M2w/S1 = 0.8880 against the 0.8750 corrected BAND ceiling -- above it by "
    "+0.0130 -- and doc 47 SS18.1 attributes that to the population mismatch, "
    "not to a broken bound. Those confirm numbers live in doc 47 SS18 and in no "
    "committed results_*.json, so they are NOT rows in this table.",
    "exp62's bounds are CONSTRUCTIVE LOWER bounds from a budgeted search over "
    "the actual held-out noun angles of their own draw. Ratios above 1.0 there "
    "mean the search did not find the optimum, not that the circuit beat "
    "capacity.",
    "exp65's and exp67's bounds are DOC-REGISTERED: computed in sessions whose "
    "code was never committed. They are carried with that provenance and are "
    "not re-derived by this script. exp67's is empirically corroborated by its "
    "classical comparator landing on it (doc 53 comparator meta_notes).",
    "exp65's bound of 1.0000 cannot test a cap: doc 51 SS11.3 records that "
    "15.8% of random 3-vs-5 splits of its 8 held-out words also reach 1.0000.",
    "doc 54 SS7 read NO exp68 criterion against a bound, and doc 52 SS2.2 "
    "removed a 9-dim-per-referent bound's standing as an instrument on a "
    "24-noun inventory. exp68's rows are validation of the INSTRUMENT, not a "
    "re-reading of exp68's verdicts.",
    "Every arm here is a simulation of a product-state encoding whose kernel "
    "doc 52 SS1 proved factorises and is classically O(n). Nothing in this "
    "table is a quantum-advantage claim, and the classical-comparator rows are "
    "in the figure precisely so that it cannot be read as one.",
]


def build(strict=False):
    rows, R = assemble()
    flagged = [r for r in rows if r["over_bound_flag"]]
    over = [r for r in rows if r["over_bound"] and not r["over_bound_flag"]]
    payload = {
        "instrument": "capacity_validation.py",
        "script_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "capacity_check_sha256": hashlib.sha256(
            open(os.path.join(HERE, "capacity_check.py"), "rb").read()).hexdigest(),
        "instrument_sb_bounds_sha256": isb.self_sha256(),
        "what_this_is": ("prediction vs measurement for the capacity instrument. "
                         "ANALYSIS OF FROZEN ARTIFACTS ONLY -- no training, no "
                         "new experiment, no existing file modified."),
        "not_a_results_file": ("named diagnostic_*, not results_*: it is an "
                               "instrument output, not an experiment result"),
        "source_files": R.files,
        "family_rule": FAMILY_RULE,
        "subfamily_rule": SUBFAMILY,
        "over_bound_margin": OVER_BOUND_MARGIN,
        "cross_checks": {
            "n_checks": len(R.checks), "n_failed": len(R.failures),
            "all_passed": not R.failures, "failed": R.failures,
            "checks": R.checks,
            "discipline": ("every value in `rows` is re-read from its source "
                           "file at runtime; the expectations above are "
                           "hand-typed IN THIS SCRIPT so that a drift in any "
                           "committed artifact fails loudly"),
        },
        "selftest_capacity_check": capacity_check.selftest(verbose=False),
        "n_rows": len(rows),
        "rows_by_role": {k: sum(1 for r in rows if r["role"] == k)
                         for k in ("primary", "context", "floor")},
        "family_summary": summarise(rows),
        "family_summary_primary_only": summarise(rows, primary_only=True),
        "conversion_by_bound_population": _by_population(rows),
        "over_bound_flagged_gt_margin": [
            {"experiment": r["experiment"], "arm": r["arm"], "cell": r["cell"],
             "bound": r["bound"], "measured": r["measured_mean"],
             "gap": r["gap"]} for r in flagged],
        "over_bound_within_margin": [
            {"experiment": r["experiment"], "arm": r["arm"], "cell": r["cell"],
             "bound": r["bound"], "measured": r["measured_mean"],
             "gap": r["gap"], "why": r["bound_provenance"]["kind"]}
            for r in over],
        "caveats": CAVEATS,
        "rows": rows,
    }
    if strict and R.failures:
        raise AssertionError("cross-checks failed: %s" % R.failures)
    return payload, rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)

    payload, rows = build(strict=a.strict)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote %s  (%d rows: %s)" % (a.out, len(rows), payload["rows_by_role"]))
    print("cross-checks: %d/%d passed" % (payload["cross_checks"]["n_checks"]
                                          - payload["cross_checks"]["n_failed"],
                                          payload["cross_checks"]["n_checks"]))
    for f in payload["cross_checks"]["failed"]:
        print("  FAILED: %s" % f)
    print("capacity_check selftest: %s"
          % ("PASS" if payload["selftest_capacity_check"]["all_passed"] else "FAIL"))
    print("conversion ratio by BOUND POPULATION (primary rows):")
    for k, v in payload["conversion_by_bound_population"].items():
        print("  %-42s n %2d  med %.3f  mean %.3f"
              % (k, v["n"], v["median"], v["mean"]))
    print("conversion ratio by family (primary + context, excl. floors):")
    for k, v in payload["family_summary"].items():
        print("  %-22s rows %2d (primary %2d) ratio min %s med %s max %s"
              % (k, v["n_rows"], v["n_primary"],
                 "n/a" if v["conversion_ratio_min"] is None else "%.3f" % v["conversion_ratio_min"],
                 "n/a" if v["conversion_ratio_median"] is None else "%.3f" % v["conversion_ratio_median"],
                 "n/a" if v["conversion_ratio_max"] is None else "%.3f" % v["conversion_ratio_max"]))
    if payload["over_bound_flagged_gt_margin"]:
        print("*** MEASURED ABOVE BOUND BY MORE THAN %.2f ***" % OVER_BOUND_MARGIN)
        for r in payload["over_bound_flagged_gt_margin"]:
            print("   %s %s %s: bound %.4f measured %.4f (+%.4f)"
                  % (r["experiment"], r["arm"], r["cell"], r["bound"],
                     r["measured"], r["gap"]))
    else:
        print("no row exceeds its bound by more than %.2f" % OVER_BOUND_MARGIN)
    for r in payload["over_bound_within_margin"]:
        print("   over bound (within margin): %s %s %s bound %.4f measured %.4f "
              "(+%.4f, %s bound)" % (r["experiment"], r["arm"], r["cell"],
                                     r["bound"], r["measured"], r["gap"], r["why"]))

    if not a.no_figure:
        try:
            import capacity_validation_figure as fig
        except ImportError:
            print("figure module not present; skipping (--no-figure to silence)")
        else:
            fig.render(rows, FIG_SVG, FIG_PNG, FIG_DATA)
            print("wrote %s, %s, %s" % (FIG_SVG, FIG_PNG, FIG_DATA))
    return 0 if payload["cross_checks"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
