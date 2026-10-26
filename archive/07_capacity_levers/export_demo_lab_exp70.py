# -*- coding: utf-8 -*-
"""export_demo_lab_exp70.py -- distil the committed exp70 artefacts into ONE
small bundle, demo/exp70_lab.json, for the language-lab page.

WHY A SEPARATE EXPORTER, AND NOT A NEW SECTION OF export_demo_lab.py.
Doc 56's own Rule-3 discipline: the exp70 anchors write their OWN results file
and touch no exp69 artefact.  The same rule is worth keeping here.
export_demo_lab.py regenerates demo/model_exp69.json, demo/testset_lab.json and
demo/featurizer_lab.json from torch checkpoints; re-running it to add an exp70
payload would put three committed exp69 assets at risk for no benefit.  This
script writes exactly one new file and reads five committed JSONs read-only.

NO LIVE exp70 MODEL IS SHIPPED.  The page renders exp70 from committed numbers
only.  Running an exp70 arm in the browser would need the CRY intro in
engine_lab.js, an eps-map angle table, an exp70 held-out testset AND an
extension of verify_demo_parity_lab.py to re-derive all of it against PyTorch.
Until that gate exists there is no honest way to put an exp70 circuit on screen,
and an ungated live model is not an option.  So: numbers of record, cited.

WHAT IS GATED HERE, rather than trusted:
  1. Every source file's sha256 is recomputed and written into the bundle, so
     the page can print the provenance it is actually rendering.
  2. The E-in concurrence CLOSED FORM |sin t1| * |sin(t2/2)| is re-evaluated
     from the committed (p1, p2) coordinates through both angle maps and must
     reproduce diagnostic_encoding_entanglement.json's per-noun concurrences
     exactly (0.0 absolute difference on all 24 nouns).  The page states that
     closed form; this is what entitles it to.
  3. The eps-map dead-zone count, the ladder ordering, the comparator identity
     (comparator_minus_circuit == comparator mean - arm mean) and the anchor
     bit-exactness are re-derived from the arrays rather than copied from the
     summary booleans.
  4. NO PRIVATE PATH MAY LEAVE THIS SCRIPT.  Every string written is passed
     through pub(), and the finished bundle is scanned: a single occurrence of
     a private-repo path or a local filesystem path refuses the write.

If any of that fails, nothing is written.
"""
import hashlib
import json
import math
import os
import re
import sys

RESULTS = "results_exp70.json"
PREDICTIONS = "exp70_predictions.json"
DIAG_ENT = "diagnostic_encoding_entanglement.json"
DIAG_BOUNDS = "diagnostic_entangled_bounds.json"
ANCHORS = "results_exp70_anchors.json"
OUT_DIR = "demo"
OUT = os.path.join(OUT_DIR, "exp70_lab.json")

SOURCES = [
    (RESULTS, "the results of record -- every arm mean, floor, comparator tier, "
              "criterion boolean and composition cost the page prints"),
    (PREDICTIONS, "the registered predictions, written BEFORE exp70_run.py existed"),
    (DIAG_ENT, "the pre-registration degeneracy check -- per-noun concurrence "
               "under both angle maps"),
    (DIAG_BOUNDS, "the grade-1 bounds instrument -- the closed forms, the spans "
                  "and the Clifford certification"),
    (ANCHORS, "the blocking replication gate against exp69"),
]

ARM_ORDER = ["A0-W-T1", "A0-W-T2", "EA-ctrl-T1", "EA-ctrl-T2",
             "P-eps-T1", "P-eps-T2", "N-in-T1", "N-in-T2",
             "N-across-T1", "N-across-T2"]

# The five-rung composition-fragility ladder, cheapest first.  The KEYS are the
# committed cc_* keys; no value is typed here.  The labels are prose.
LADDER = [
    ("cc_EA-ctrl", "one angle per word (plain)", "زاوية واحدة لكل كلمة",
     "plain angle", "زاوية واحدة",
     "one RY per referent -- the smallest encoding in the programme"),
    ("cc_N-across", "one angle, then tangle the two words together",
     "زاوية واحدة ثم تشبيك الكلمتين",
     "cross-tangle", "تشابك متقاطع",
     "the same single angle, plus one CZ ACROSS the two referents"),
    ("cc_P-eps", "two numbers per word, better map",
     "رقمان لكل كلمة، خريطة أفضل",
     "register, new map", "سجل، خريطة جديدة",
     "a two-qubit product register on the eps-map"),
    ("cc_A0-W", "two numbers per word, old map",
     "رقمان لكل كلمة، الخريطة القديمة",
     "register, old map", "سجل، خريطة قديمة",
     "the same product register on exp68's registered map -- exp69's R-W arm"),
    ("cc_N-in", "two numbers per word, tangled at birth",
     "رقمان لكل كلمة، مشبوكان منذ البداية",
     "tangled at birth", "مشبوك منذ البداية",
     "the entangled register: RY then CRY, the word tangled with itself"),
]

# ---------------------------------------------------------------- hygiene
# The committed JSONs cite the design doc by its path inside a PRIVATE repo, and
# a demo page may cite public file names only.  Rewrites, not deletions: the
# reference survives, the path does not.
_PRIVATE = [
    (re.compile(r"qnlp_private_docs/\S*?56_exp70_entangled_encodings[A-Za-z_]*\.md"
                r"(\s*\([^)]*\))?"), "doc 56"),
    (re.compile(r"qnlp_private_docs/\S+\.md"), "the registered design doc"),
    (re.compile(r"qnlp_private_docs\S*"), "the registered design doc"),
]
_SS = re.compile(r"\bSS(?=\d)")
_FORBIDDEN = ("qnlp_private_docs", "/Users/", "/home/", "\\Users\\")


def pub(s):
    """Publishable form of a committed prose string: private paths rewritten to
    a doc citation, the ASCII 'SSn' section marker rendered as the section sign
    the rest of the page uses."""
    if not isinstance(s, str):
        return s
    for rx, rep in _PRIVATE:
        s = rx.sub(rep, s)
    s = _SS.sub("§", s)
    # the committed files write an ASCII double hyphen where this page's prose
    # uses an em dash; purely typographic, applied so quoted registered wording
    # does not read as foreign matter next to the page's own sentences
    s = s.replace(" -- ", " — ")
    # a few committed strings emphasise with markdown; this page renders escaped
    # text, so the asterisks would show. Dropped, not reinterpreted -- the words
    # they wrapped are unchanged.
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return s


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def die(msg):
    print("[export70] REFUSING: " + msg)
    sys.exit(1)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- the maps
# Both angle maps, as the committed diagnostic states them.  These are here to
# be CHECKED against the committed per-noun numbers, not to replace them.
EPS = math.pi / 6


def eps_map(p):
    return p * (math.pi - 2 * EPS) / 2 + math.pi / 2


def registered_map(p):
    return (p + 1) * math.pi


def concurrence(t1, t2):
    return abs(math.sin(t1)) * abs(math.sin(t2 / 2))


def main():
    for f, _ in SOURCES:
        if not os.path.exists(f):
            die("missing committed source %s" % f)
    R = json.load(open(RESULTS, encoding="utf-8"))
    P = json.load(open(PREDICTIONS, encoding="utf-8"))
    D = json.load(open(DIAG_ENT, encoding="utf-8"))
    B = json.load(open(DIAG_BOUNDS, encoding="utf-8"))
    A = json.load(open(ANCHORS, encoding="utf-8"))

    if list(R["arms"].keys()) != ARM_ORDER:
        die("results_exp70.json arm order is not the registered order")

    # ============================================== GATE 1: the closed form
    # The page states concurrence = |sin t1| . |sin(t2/2)| as a closed form.
    # Re-evaluate it from the committed (p1, p2) through both maps and require
    # it to reproduce the committed per-noun concurrences exactly.
    per_noun, worst_eps, worst_reg = [], 0.0, 0.0
    for noun, e in D["per_noun"].items():
        ce = concurrence(eps_map(e["p1"]), eps_map(e["p2"]))
        cr = concurrence(registered_map(e["p1"]), registered_map(e["p2"]))
        worst_eps = max(worst_eps, abs(ce - e["concurrence_eps_map"]))
        worst_reg = max(worst_reg, abs(cr - e["concurrence_registered_map"]))
        per_noun.append({
            "noun": noun,
            "p1": e["p1"], "p2": e["p2"],
            "concurrence_registered_map": e["concurrence_registered_map"],
            "concurrence_eps_map": e["concurrence_eps_map"],
            "frozen_angle": e["frozen_angle"],
        })
    if worst_eps != 0.0 or worst_reg != 0.0:
        die("the concurrence closed form does not reproduce the committed "
            "per-noun values exactly (eps %r, registered %r)"
            % (worst_eps, worst_reg))
    per_noun.sort(key=lambda r: r["concurrence_registered_map"])

    # ============================================== GATE 2: the dead zone
    dead = [r for r in per_noun if r["concurrence_registered_map"] < 0.05]
    if len(dead) != D["finding"]["registered_map"]["n_below_0.05"]:
        die("recounted dead-zone nouns (%d) disagree with the committed count (%d)"
            % (len(dead), D["finding"]["registered_map"]["n_below_0.05"]))
    if any(r["concurrence_eps_map"] < 0.05 for r in per_noun) or \
            D["finding"]["eps_map"]["n_below_0.05"] != 0:
        die("the eps-map is supposed to have no dead zone and does")
    # the on-page example: the noun whose |p1| sits closest to the normalisation
    # extreme -- chosen by the data, not by hand
    example = min(per_noun, key=lambda r: abs(abs(r["p1"]) - 1.0))

    # ============================================== GATE 3: the ladder
    cc = R["composition_cost_doc56_SS6_7"]["per_family"]
    ladder = []
    for key, en, ar, short_en, short_ar, sub in LADDER:
        if key not in cc:
            die("no committed composition cost for %s" % key)
        c = cc[key]
        ladder.append({"key": key, "label_en": en, "label_ar": ar,
                       "short_en": short_en, "short_ar": short_ar, "detail": sub,
                       "pair": c["pair"], "mean": c["mean"], "sd": c["sd"],
                       "per_seed": c["per_seed"]})
    vals = [r["mean"] for r in ladder]
    if vals != sorted(vals):
        die("the ladder is not monotone in the committed cc values -- the page's "
            "'staircase' reading would be false: %r" % vals)
    # the exp69 replication cross-note: exp70's own A0/EA-ctrl costs must land on
    # exp69's committed cc_A / cc_W to the precision exp69 published them at
    ref = R["composition_cost_doc56_SS6_7"]["exp69_reference_values"]
    if not close(cc["cc_EA-ctrl"]["mean"], ref["cc_A"], 5e-4) or \
            not close(cc["cc_A0-W"]["mean"], ref["cc_W"], 5e-4):
        die("exp70's anchor composition costs do not land on exp69's published "
            "cc_A / cc_W")

    # ============================================== GATE 4: the arms
    arms = {}
    for name in ARM_ORDER:
        a = R["arms"][name]
        fl = R["floors"].get(name)
        fc = R["floor_clearance"].get(name)
        fam = a["family"]
        tiers = R["comparator"]["tiers_per_family"][fam]
        comp = {}
        for t in tiers:
            c = R["comparator"]["arms"][name][t]
            delta = R["comparator"]["comparator_minus_circuit"][name][t]
            if not close(delta, c["mean"] - a["mean"], 1e-9):
                die("comparator_minus_circuit disagrees with the means for "
                    "%s / %s" % (name, t))
            comp[t] = {"mean": c["mean"], "sd": c["sd"],
                       "feature_dim": c["feature_dim"],
                       "minus_circuit": delta,
                       "tier_role": pub(c["tier_role"]),
                       "feature_definition": pub(c["feature_definition"])}
        arb = R["comparator"]["arbiter_doc56_SS8_4"].get(name)
        bnd = R["bounds_context_only_doc56_SS10"][name]
        arms[name] = {
            "arm": name, "T": a["T"], "family": fam, "encoding": a["encoding"],
            "role": pub(a["role"]),
            "mean": a["mean"], "sd": a["sd"], "per_seed": a["per_seed"],
            "wires": a["wires"], "n_params": a["n_params"], "n_test": a["n_test"],
            "floor_present": fl is not None,
            "floor_mean": (fl or {}).get("mean"),
            "floor_sd": (fl or {}).get("sd"),
            "floor_sd_units": (fc or {}).get("sd_units_over_floor_using_floor_sd"),
            "clears_floor": (fc or {}).get("clears_floor"),
            "clears_under_all_readings": (fc or {}).get("clears_under_all_readings"),
            "binds": R["verdicts"].get("binds_%s_T%d" % (fam, a["T"])),
            "bound_of_record": bnd["bound_of_record"],
            "over_bound_by": bnd["over_bound_by"],
            "tiers": tiers,
            "comparator": comp,
            "arbiter_row": pub(arb["row"]) if arb else None,
            "arbiter_reading": pub(arb["reading"]) if arb else None,
        }
    # the anchor arms carry NO floor by the registered floor-omission argument;
    # the three new families all do. Asserted, not assumed.
    if any(arms[n]["floor_present"] for n in ARM_ORDER if
           arms[n]["family"] in ("A0-W", "EA-ctrl")):
        die("an anchor arm unexpectedly carries its own floor")
    if not all(arms[n]["floor_present"] for n in ARM_ORDER if
               arms[n]["family"] in ("P-eps", "N-in", "N-across")):
        die("a new-family arm is missing its floor")

    # ============================================== GATE 5: the anchors
    anchors = {}
    for name, blk in A["per_arm"].items():
        recomputed = max(abs(x - y) for x, y in
                         zip(blk["exp69_per_seed"], blk["exp70_per_seed"]))
        if (recomputed == 0.0) != bool(blk["bit_exact"]):
            die("the anchor bit-exact flag for %s disagrees with its own arrays"
                % name)
        anchors[name] = {
            "exp69_arm": blk["exp69_arm"],
            "exp69_mean": blk["exp69_mean"], "exp70_mean": blk["exp70_mean"],
            "exp69_per_seed": blk["exp69_per_seed"],
            "exp70_per_seed": blk["exp70_per_seed"],
            "max_abs_per_seed_delta": blk["max_abs_per_seed_delta"],
            "max_abs_per_seed_delta_recomputed": recomputed,
            "bit_exact": blk["bit_exact"], "replicates": blk["replicates"],
        }
    if not (A["all_bit_exact"] and A["anchors_replicate_exp69"]):
        die("the committed anchor gate is not green -- there is nothing to show")

    # ============================================== the one-sided criteria
    V = R["verdicts"]
    one_sided_keys = ["entanglement_helps_T1", "entanglement_hurts_T1",
                      "entanglement_helps_T2", "entanglement_hurts_T2",
                      "across_helps_T1", "across_hurts_T1",
                      "across_helps_T2", "across_hurts_T2"]
    booleans = []
    for k in one_sided_keys:
        ev = V["expectation_vs_outcome"][k]
        booleans.append({"key": k, "measured": V[k],
                         "registered": pub(ev["registered"]),
                         "as_registered": ev["as_registered"]})
    n_as_registered = sum(1 for b in booleans if b["as_registered"])
    if n_as_registered != len(booleans):
        die("a one-sided boolean came back off-registration and the page's "
            "wording assumes none did")
    pairs = {}
    for k, v in V["one_sided_pairs_doc56_SS6_4"].items():
        pairs[k] = {"pair": v["pair"], "delta": v["delta"], "bar": v["bar"],
                    "both_arms_gated": v["both_arms_gated"],
                    "comparator_gap_shared_tier": v["comparator_gap_shared_tier"],
                    "shared_tiers": v["shared_tiers"],
                    "comparator_clause_blocks": v["comparator_clause_blocks"],
                    "helps": v["helps"], "hurts": v["hurts"], "tag": v["tag"],
                    "reading": pub(V["third_outcome_reading_doc56_SS6_4"][k])}

    dc = V["the_one_directional_call_doc56_SS9_4"]
    if dc["boolean"] != "entanglement_hurts_T2" or not dc["outcome"]:
        die("the directional call is not the one the page describes")

    # the confirm-draw rider that must travel with the headline, taken from the
    # outcome row that carries it rather than paraphrased
    headline_row = R["outcome_rows_doc56_SS11"][0]

    # ============================================== the bundle
    out = {
        "component": "exp70_lab -- the committed exp70 payload the language-lab "
                     "page renders. NO LIVE exp70 MODEL IS BUNDLED: this page "
                     "shows exp70 as numbers of record, because no browser-side "
                     "parity gate exists for the CRY intro.",
        "exporter": os.path.basename(__file__),
        "experiment": pub(R["experiment"]),
        "question": pub(R["question"]),
        "spec_citation": "doc 56, REGISTERED 2026-08-25, spec commit "
                         + R["spec_commit"],
        "spec_citation_note": "the design doc lives in a private repository and "
                              "is cited by number and commit only; every file "
                              "named on this page is a public artefact of this "
                              "repository.",
        "sources": [{"file": f, "sha256": sha256_file(f), "role": role}
                    for f, role in SOURCES],
        "n_seeds": R["n_seeds"], "n_runs": R["n_runs"],
        "sd_convention": R["sd_convention"],
        "arm_order": ARM_ORDER,
        "arms": arms,
        "criterion_bars": {
            "delta_bar": V["criterion_bars"]["delta_bar"],
            "binds_absolute": V["criterion_bars"]["binds_absolute"],
            "floor_sd_multiple": V["criterion_bars"]["floor_sd_multiple"],
            "anchor_tolerance": V["criterion_bars"]["anchor_tolerance"],
            "threshold_provenance": pub(V["criterion_bars"]["threshold_provenance"]),
        },
        "ladder": {
            "rows": ladder,
            "status": pub(R["composition_cost_doc56_SS6_7"]["status"]),
            "no_promotion_clause":
                pub(R["composition_cost_doc56_SS6_7"]["no_promotion_clause"]),
            "population_note":
                pub(R["composition_cost_doc56_SS6_7"]["population_note"]),
            "exp69_reference_values": ref,
            "definition": "composition cost cc = (the family's T = 1 held-out "
                          "mean) - (its T = 2 held-out mean), 5 seeds, from "
                          + RESULTS,
        },
        "one_sided": {
            "booleans": booleans,
            "n_as_registered": n_as_registered,
            "pairs": pairs,
            "directional_call": {
                "boolean": dc["boolean"],
                "registered_expectation": dc["registered_expectation"],
                "outcome": dc["outcome"],
                "delta_measured": dc["delta_measured"],
                "bar": dc["bar"],
                "verdict": pub(dc["verdict"]),
                "why_it_was_registered": pub(dc["why_it_was_registered"]),
                "reading": pub(dc["reading"]),
            },
            "headline_row": {k: pub(v) for k, v in headline_row.items()},
        },
        "e_in": {
            "form": pub(D["e_in_form"]),
            "state": B["derivations"]["E_in"]["state"],
            "amplitude_on_01": B["derivations"]["E_in"]["amplitude_on_01"],
            "concurrence": B["derivations"]["E_in"]["concurrence"],
            "per_wire_purity": B["derivations"]["E_in"]["per_wire_purity"],
            "span": B["derivations"]["E_in"]["span"],
            "d_per_referent": B["derivations"]["E_in"]["d_per_referent"],
            "d_joint": B["derivations"]["E_in"]["d_joint"],
            "relation_to_the_product_register":
                pub(B["derivations"]["E_in"]["relation_to_the_product_register"]),
            "completeness": pub(B["derivations"]["E_in"]["completeness"]),
            "closed_form_recheck": {
                "max_abs_error_eps_map": worst_eps,
                "max_abs_error_registered_map": worst_reg,
                "n_nouns": len(per_noun),
                "note": "re-evaluated in " + os.path.basename(__file__)
                        + " from the committed (p1, p2) coordinates through both "
                        "angle maps; the export refuses to write unless it "
                        "reproduces " + DIAG_ENT + " exactly.",
            },
        },
        "maps": {
            "registered_map": pub(D["registered_map"]),
            "eps_map": pub(D["eps_map"]),
            "eps": B["eps_map"]["eps"],
            "eps_formula": B["eps_map"]["formula"],
            "eps_range": B["eps_map"]["range"],
            "forced_by": pub(B["eps_map"]["forced_by"]),
            "design_consequence": pub(D["design_consequence"]),
        },
        "dead_zone": {
            "registered_map_stats": D["finding"]["registered_map"],
            "eps_map_stats": D["finding"]["eps_map"],
            "mechanism": pub(D["finding"]["registered_map"]["mechanism"]),
            "dead_nouns": dead,
            "example": example,
            "per_noun": per_noun,
            "gate_min_satisfied":
                B["concurrence_and_purity"]["e_in_eps_map"]["gate_min_0.10_satisfied"],
        },
        "e_across": {
            "state": B["derivations"]["E_across"]["state"],
            "concurrence": B["derivations"]["E_across"]["concurrence"],
            "per_wire_purity": B["derivations"]["E_across"]["per_wire_purity"],
            "d_joint": B["derivations"]["E_across"]["d_joint"],
            "effective_rank": B["derivations"]["E_across"]["effective_rank"],
            "clifford_argument": pub(B["derivations"]["E_across"]["clifford_argument"]),
            "consequences": [pub(c) for c in
                             B["derivations"]["E_across"]["consequences"]],
            "completeness": pub(B["derivations"]["E_across"]["completeness"]),
            "statement": pub(P["clifford_result_doc56_SS4_5"]["statement"]),
            "the_finding": pub(R["meta_notes"]["the_clifford_finding_doc56_SS4_5"]),
            "clifford_consequence": pub(R["comparator"]["clifford_consequence"]),
            "weak_by_construction":
                pub(R["meta_notes"]["e_across_weak_by_construction_doc56_SS18_4"]),
            "heldout_entanglement":
                R["meta_notes"]["heldout_entanglement_measured"]["e_across"],
            "intro_purity": D["e_across_intro_purity"],
        },
        "anchors": {
            "per_arm": anchors,
            "tolerance": A["tolerance"],
            "all_bit_exact": A["all_bit_exact"],
            "anchors_replicate_exp69": A["anchors_replicate_exp69"],
            "dataset": A["dataset"], "dataset_sha256": A["dataset_sha256"],
            "exp69_source": A["exp69_source"],
            "honest_expectation": pub(A["honest_expectation"]),
            "consequence_if_failed": pub(A["consequence_if_failed"]),
        },
        "comparator": {
            "tiers_per_family": R["comparator"]["tiers_per_family"],
            "reading_rule": pub(R["comparator"]["reading_rule"]),
            "two_tier_reading": {k: {kk: pub(vv) if isinstance(vv, str) else vv
                                     for kk, vv in v.items()}
                                 for k, v in
                                 R["comparator"]["two_tier_reading_doc56_SS8_1"].items()},
            "matches_or_exceeds_all_arms":
                V["comparator_matches_or_exceeds_all_arms"],
            "matches_or_exceeds_all_arms_STRICT_every_tier":
                V["comparator_matches_or_exceeds_all_arms_STRICT_every_tier"],
            "why_the_outer_square": pub(R["comparator"]["why_the_outer_square"]),
            "function_class_qualifier":
                pub(R["comparator"]["function_class_qualifier"]),
        },
        "outcome_rows": [{k: pub(v) for k, v in row.items()}
                         for row in R["outcome_rows_doc56_SS11"]],
        "scope": [pub(s) for s in R["meta_notes"]["scope_doc56_SS12"]],
        "no_datagen": pub(R["no_datagen_doc56_SS5_1"]),
        "no_wazn_variant": pub(R["meta_notes"]["no_wazn_variant_doc56_SS5_2"]),
        "bounds_are_context_only":
            pub(R["predictions_vs_measured_doc56_SS9"]["bounds_are_context_only"]),
        "no_criterion_reads_a_bound":
            pub(V["criterion_bars"]["no_criterion_reads_a_bound"]),
    }

    # ============================================== GATE 6: nothing private
    blob = json.dumps(out, ensure_ascii=False)
    hits = [t for t in _FORBIDDEN if t in blob]
    if hits:
        die("the bundle carries a private or local path: %r" % hits)

    if not os.path.isdir(OUT_DIR):
        die("no %s directory" % OUT_DIR)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=False)
    print("[export70] wrote %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))
    print("[export70] gate 1  concurrence closed form reproduces %s on %d/%d "
          "nouns, max |d| = %g (eps map) / %g (registered map)"
          % (DIAG_ENT, len(per_noun), len(per_noun), worst_eps, worst_reg))
    print("[export70] gate 2  eps-map dead zone: 0 nouns below 0.05; registered "
          "map: %d (%s)"
          % (len(dead), "، ".join(r["noun"] for r in dead)))
    print("[export70] gate 3  ladder monotone: "
          + " -> ".join("%s %.4f" % (r["key"], r["mean"]) for r in ladder))
    print("[export70] gate 4  10 arms, comparator tiers consistent, floors "
          "present on the 3 new families and absent on the 2 anchor families")
    print("[export70] gate 5  anchors bit-exact on %d arms (max |per-seed d| = "
          "%g)" % (len(anchors),
                   max(a["max_abs_per_seed_delta_recomputed"]
                       for a in anchors.values())))
    print("[export70] gate 6  no private-repo or local path in the bundle")
    for s in out["sources"]:
        print("[export70]   %-42s %s" % (s["file"], s["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
