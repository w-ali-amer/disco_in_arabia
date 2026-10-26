# -*- coding: utf-8 -*-
"""export_demo_lab.py -- emit the assets for the exp69 "language lab" page.

Track 3 of the demonstration phase, and a NEW namespace: nothing here reads any
exp51 or exp68 demo artefact for writing, and nothing outside `demo/*_lab*` /
`demo/model_exp69.json` / `demo/featurizer_lab.json` / `demo/testset_lab.json`
is created or touched (demo charter, Rule 3).

Same discipline as export_demo_exp68.py (demo_engine_spec.md SS5): every number
written here traces to a committed artefact, and every checkpoint is verified to
reproduce its committed per-seed held-out accuracy BEFORE anything is written.

  demo/model_exp69.json      six trained arms (R1-A/W/M, R2-A/W/M) at the
                             val-selected seed, the frozen per-noun angle table
                             for all three encodings, the committed comparator /
                             bound / purity / kernel numbers the page displays.
  demo/featurizer_lab.json   the FITTED exp65 positional featuriser basis --
                             char-position vocabulary, centring vector, the 2-row
                             projection -- so the browser can compute a wazn
                             angle for a word the visitor TYPES.
  demo/testset_lab.json      the four committed held-out splits
                             (QA x T1, QA x T2, QB x T1, QB x T2), 300 items each,
                             compiled fields only, with twin structure.

SEED SELECTION -- READ THIS.  `exp69_aggregate.py` performs NO per-arm seed
selection: it reports 5-seed means, and its "model_selection: best
in-distribution val accuracy" refers to the per-EPOCH checkpoint choice inside
one run.  A demo must run ONE model per arm, so a selection rule is required and
is stated here explicitly rather than implied: argmax over the committed
`val_per_seed`, ties broken by the LOWEST seed index.  Selection uses
in-distribution val only and never peeks at held-out.  Recorded verbatim in the
exported JSON.  (Identical rule and identical wording to export_demo_exp68.py --
inherited on purpose so the two pages cannot drift apart.)

THE FEATURISER EXTRACTION IS GATED, NOT TRUSTED.  `exp65_featurizer.fit_featurizer`
hides its vocabulary, centring vector and PCA basis inside a closure, so this
script re-fits through the same code path and extracts them.  Before anything is
written, the extracted basis must reproduce
`exp65_featurizer.angles_for(all 24 humans, all 24 humans)` -- the INVENTORY-WIDE
fit doc 55 SS18.4 registers for exp69 -- to 0.0 absolute difference on all 24
nouns.  If it does not, the export refuses.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

import exp41_story_generator as gen
import exp42_compiler as comp
import exp65_featurizer as WZ
import exp69_run as R
import instrument_sb_bounds as INS

RESULTS = "results_exp69.json"
PREDICTIONS = "exp69_predictions.json"
KERNEL = "diagnostic_kernel_factorisation.json"
RAW = "aravec_raw_vectors_44.json"
DIAG_SB = "diagnostic_sb_bounds.json"
CKPT_DIR = "exp69_ckpt"
OUT = "demo"

ARMS = ["R1-A", "R1-W", "R1-M", "R2-A", "R2-W", "R2-M"]

SELECTION_RULE = ("argmax over results_exp69.json arms[<arm>].val_per_seed "
                  "(in-distribution validation accuracy of the selected "
                  "checkpoint); ties broken by the LOWEST seed index. "
                  "Held-out accuracy is never consulted. NOTE: "
                  "exp69_aggregate.py itself selects no seed -- it reports "
                  "5-seed means -- so this rule is stated here, not inherited.")

# --------------------------------------------------------------------------
# Demo input hygiene.  NOT part of the frozen featuriser (which does only
# `strip_al`): this is the normalisation the PAGE applies to a word a visitor
# types, before the frozen featuriser sees it.  It is applied identically on
# both sides of the parity gate, and is declared in featurizer_lab.json so the
# page can state it on screen.
TATWEEL = "ـ"
HARAKAT = ("ًٌٍَُِّْٕٓٔ"
           "ٰٖٜٟٗ٘ٙٚٛٝٞ")


def normalize_input(w):
    """Demo input hygiene: drop tatweel and Arabic diacritics, collapse
    whitespace.  Mirrored character-for-character in demo/featurizer_lab.js."""
    out = [c for c in w if c != TATWEEL and c not in HARAKAT]
    return "".join(out).strip()


# The REGISTERED novel-word list for the featuriser parity gate.  Fixed here,
# in the export, so the gate cannot be tuned after seeing its result.  Ten
# faaC/CaaCiC-pattern words and ten that are not; one carries diacritics and one
# carries tatweel, so the normalisation path is exercised end to end.
NOVEL_WORDS = [
    # --- faail (CaaCiC) pattern, none of them in the 24-noun inventory
    ("الشاعر", "poet", True),
    ("القاضي", "judge", True),
    ("الراكب", "rider", True),
    ("الحاكم", "ruler", True),
    ("السامع", "listener", True),
    ("البائع", "seller", True),
    ("الزائر", "visitor", True),
    ("الناجح", "successful one", True),
    ("العازف", "musician", True),
    ("الكــاتم", "concealer (TATWEEL)", True),
    # --- not faail
    ("الكتاب", "the book", False),
    ("المدرسة", "the school", False),
    ("البيت", "the house", False),
    ("مكتوب", "written", False),
    ("استخراج", "extraction", False),
    ("تعليم", "education", False),
    ("الطويل", "the tall one", False),
    ("مستشفى", "hospital", False),
    ("قِطَار", "train (DIACRITICS)", False),
    ("المكتبة", "the library", False),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def select_seed(block):
    """(seed, index) of the val-selected run.  Deterministic; see SELECTION_RULE."""
    pairs = list(zip(block["seeds"], block["val_per_seed"]))
    best = max(pairs, key=lambda sv: (sv[1], -sv[0]))
    idx = block["seeds"].index(best[0])
    return best[0], idx


# ======================================================== the featuriser basis
def wazn_tok(word):
    """`exp65_featurizer.fit_featurizer.tok`, character for character.  Copied
    rather than imported because it is defined inside the fit closure; the
    equality gate below is what makes the copy safe."""
    c = list(WZ.strip_al(word))
    return " ".join("%d:%s" % (i, ch) for i, ch in enumerate(c)) + " LEN%d" % len(c)


def fit_wazn_basis(train_nouns):
    """`exp65_featurizer.fit_featurizer` with the closure opened up.  Same
    CountVectorizer configuration, same centring, same `Vt[:2]`."""
    vec = CountVectorizer(analyzer="word", token_pattern=r"\S+")
    X = vec.fit_transform([wazn_tok(w) for w in train_nouns]).toarray().astype(float)
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return vec, mu[0], Vt[:2]


def angle_from_basis(word, vocab, mu, basis):
    """The exported basis, applied exactly as `angle_of` applies the closure."""
    x = np.zeros(len(mu))
    for t in wazn_tok(word).split():
        j = vocab.get(t.lower())
        if j is not None:
            x[j] += 1.0
    p = (x - mu) @ basis.T
    n = float(np.linalg.norm(p))
    if n < 1e-12:
        return math.pi
    p = p / n
    return float(math.atan2(p[1], p[0]))


# ============================================================ the arc calculus
def arc_bound(angles, labels):
    """EXACT best-2-arc bound for a d = 3 binary readout on the circle.

    Every readout of a single-RY referent state is p(theta) = a + b cos theta +
    c sin theta = a + R cos(theta - phi).  A binary decision is the sign of one
    such function, so the YES region is ONE contiguous arc and the circle is cut
    into at most 2 arcs -- exactly the M = 2 row of
    diagnostic_readout_arity_bounds.json's `exact_maximum_2M_minus_2`.  The best
    achievable per-noun accuracy is therefore the best contiguous window in
    angular order (the complement of a window is a window, so labelling the
    window YES covers both orientations).

    Returns (best_count, n, window) with `window` the (start, length) in the
    angle-sorted order.  O(n^2), and the browser runs the identical loop."""
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    y = [1 if labels[order[i]] else 0 for i in range(len(order))]
    n = len(y)
    zeros = sum(1 for v in y if v == 0)
    best, win = zeros, (0, 0)                                # the empty window
    for s in range(n):
        acc = 0
        for L in range(1, n + 1):
            acc += y[(s + L - 1) % n]
            # inside the window predict YES, outside predict NO:
            #   correct = inside_ones + outside_zeros = acc + (zeros - (L - acc))
            score = 2 * acc + zeros - L
            if score > best:
                best, win = score, (s, L)
    return best, n, win


def main():
    os.makedirs(OUT, exist_ok=True)

    res = json.load(open(RESULTS, encoding="utf-8"))
    pred = json.load(open(PREDICTIONS, encoding="utf-8"))
    kern = json.load(open(KERNEL, encoding="utf-8"))
    if not res["verdicts"]["anchor_replicates_exp68"]:
        raise RuntimeError("exp69 A0 anchor is not green -- refusing to export")
    if not res["verdicts"]["run_is_interpretable"]:
        raise RuntimeError("results_exp69.json says the run is not interpretable")

    # ---------------------------------------------------------- the datasets
    store = {}
    for part, path in (("A", R.DATA_A), ("B", R.DATA_B)):
        blob = json.load(open(path, encoding="utf-8"))
        items, meta = blob["items"], blob["meta"]
        sha = comp.canonical_sha256(items)
        if sha != meta["dataset_sha256"] or sha != res["datasets"][part]["sha256"]:
            raise RuntimeError("%s: dataset provenance FAILED" % path)
        store[part] = (items, meta, sha)
    verbs = sorted(store["A"][1]["verb_inventory"].keys())
    slots = store["A"][1]["verb_slots"]

    # -------------------------------------------------- the frozen angle maps
    fm = R.feature_maps()

    # ---- the wazn basis, extracted and GATED against the committed closure
    H = sorted(gen.HUMANS)
    vec, mu, basis = fit_wazn_basis(H)
    vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
    committed = WZ.angles_for(H, INS.HUMANS)                 # doc 55 SS18.4 fit
    mx = 0.0
    for nname in INS.HUMANS:
        mx = max(mx, abs(angle_from_basis(nname, vocab, mu, basis) - committed[nname]))
    if mx != 0.0:
        raise RuntimeError("wazn basis extraction does not reproduce "
                           "exp65_featurizer.angles_for: max abs diff %.3e" % mx)
    if max(abs(fm["theta_wazn"][n] - committed[n]) for n in INS.HUMANS) != 0.0:
        raise RuntimeError("exp69_run.feature_maps theta_wazn != committed closure")
    print("[export] wazn basis extracted: |V| = %d tokens, 24/24 angles exact "
          "(max abs diff %.1e)" % (len(vocab), mx))

    # ---- the registered novel words, computed through the COMMITTED closure
    closure = WZ.fit_featurizer(H)
    novel = []
    for raw, gloss, is_faail in NOVEL_WORDS:
        norm = normalize_input(raw)
        novel.append({"raw": raw, "normalized": norm, "gloss": gloss,
                      "faail_pattern_by_eye": is_faail,
                      "stripped": WZ.strip_al(norm),
                      "angle_python": float(closure(norm)),
                      "in_inventory": norm in set(INS.HUMANS)})
    if any(w["in_inventory"] for w in novel):
        raise RuntimeError("a 'novel' test word is in the trained inventory")

    featurizer = {
        "component": "exp65 positional featuriser, fitted basis",
        "source": "exp65_featurizer.py",
        "source_sha256": sha256_file("exp65_featurizer.py"),
        "fit_protocol": pred["grade1_bounds"]["R-M"]["featurizer_fit_protocol"],
        "fit_population": H,
        "tokenisation": ("strip the definite article ('ال' prefix), then one token "
                         "per (position, character) as '<i>:<ch>', plus a 'LEN<n>' "
                         "token; sklearn CountVectorizer(analyzer='word', "
                         "token_pattern=r'\\S+'), which lowercases (a no-op for "
                         "Arabic, so the LEN token is stored as 'len<n>')"),
        "why_positional": ("doc 51 SS11.1: faail is C-alif-C-C and faaal is "
                           "C-C-alif-C, so the alif's POSITION discriminates "
                           "them; a bag-of-characters map is blind to exactly "
                           "that."),
        "vocabulary": vocab,
        "mean": [float(v) for v in mu],
        "basis": [[float(v) for v in row] for row in basis],
        "angle_formula": ("x = counts over the fitted vocabulary; p = (x - mean) "
                          "@ basis.T; if |p| < 1e-12 -> pi, else atan2(p[1], p[0]) "
                          "after normalising p"),
        "degenerate_angle": math.pi,
        "input_hygiene": {
            "note": ("NOT part of the frozen featuriser, which does only "
                     "strip_al. This is what the PAGE does to a typed word "
                     "before the frozen featuriser sees it, and it is applied "
                     "identically on both sides of the parity gate."),
            "drop_tatweel": TATWEEL,
            "drop_diacritics": HARAKAT,
            "trim_whitespace": True,
        },
        "inventory_angles": {n: float(committed[n]) for n in INS.HUMANS},
        "inventory_angles_max_abs_diff_vs_committed_closure": mx,
        "novel_word_gate_list": novel,
        "faail_members": sorted(n for n in INS.HUMANS if n in WZ.FAAIL),
    }
    json.dump(featurizer, open(os.path.join(OUT, "featurizer_lab.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[export] wrote demo/featurizer_lab.json (%d vocab tokens, %d novel "
          "gate words)" % (len(vocab), len(novel)))

    # ------------------------------------------------------------- the arms
    arms_out = {}
    for arm in ARMS:
        blk = res["arms"][arm]
        cfg = R.ARM_CFG[arm]
        seed, idx = select_seed(blk)
        pt = os.path.join(CKPT_DIR, "%s_s%d.pt" % (arm, seed))
        sha = sha256_file(pt)
        if sha != blk["ckpt_sha256"][idx]:
            raise RuntimeError("%s: checkpoint sha mismatch %s vs committed %s"
                               % (pt, sha, blk["ckpt_sha256"][idx]))
        run = json.load(open(os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, seed)),
                             encoding="utf-8"))
        if run["ckpt_sha256"] != sha or run["seed"] != seed:
            raise RuntimeError("%s: per-run JSON disagrees with checkpoint" % pt)

        # ---- reproduce the committed per-seed held-out accuracy BEFORE export
        m = R.build_model(arm, verbs, seed, fm)
        m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        items = store[cfg["part"]][0]
        test = [i for i in items if i["split"] == "test" and i["T"] == cfg["T"]]
        ang, _t1 = R.arm_angles(arm, fm)
        cs = [R.compile_item(i, ang) for i in test]
        with torch.no_grad():
            acc = sum(int(torch.argmax(m.story_values(c)).item()) == c.answer
                      for c in cs) / len(cs)
        want = blk["per_seed"][idx]
        if abs(acc - want) > 1e-12:
            raise RuntimeError("%s s%d does not reproduce its committed held-out "
                               "accuracy: %.10f vs %.10f" % (arm, seed, acc, want))
        print("[export] %-5s T=%d part=%s enc=%-2s seed %d  val %.4f  "
              "held-out %.10f == committed  (ckpt %s)"
              % (arm, cfg["T"], cfg["part"], cfg["enc"], seed,
                 blk["val_per_seed"][idx], acc, sha[:12]))

        qpr = 2 if cfg["enc"] == "F2" else 1
        fam = R.BOUND_FAMILY[arm]
        arms_out[arm] = {
            "arm": arm,
            "T": cfg["T"],
            "part": cfg["part"],
            "encoding": cfg["enc"],
            "role": blk["role"],
            "testset_key": "%s_T%d" % (cfg["part"], cfg["T"]),
            "category_key": ("nonoccupational" if cfg["part"] == "A"
                             else "wazn_faail"),
            "model_class": ("exp68_run.F2Model (exp45_model.TwoQubitReferentModel "
                            "'A4' + the doc 54 SS6.4 second-RY override)"
                            if cfg["enc"] == "F2" else
                            "exp42_models.QuantumStoryModel 'A3'"),
            "qubits_per_referent": qpr,
            "n_qubits_per_block": 2 * qpr,
            "params_per_layer": 2 * (2 * qpr) + (2 * qpr - 1),
            "angle_source": {"FA": "phi_atan2", "F2": "theta_pc1+theta_pc2",
                             "WZ": "theta_wazn"}[cfg["enc"]],
            "verbs": verbs,
            "blocks": {v: m.blocks[v].theta.detach().tolist() for v in verbs},
            "q_pos": m.q_pos.theta.detach().tolist(),
            "q_neg": m.q_neg.theta.detach().tolist(),
            "selected_seed": seed,
            "selected_seed_index": idx,
            "selection_rule": SELECTION_RULE,
            "checkpoint": pt,
            "checkpoint_sha256": sha,
            "selected_epoch": blk["selected_epoch_per_seed"][idx],
            "val_acc_selected_seed": blk["val_per_seed"][idx],
            "train_acc_selected_seed": blk["train_per_seed"][idx],
            "heldout_acc_selected_seed": want,
            "heldout_acc_recomputed_at_export": acc,
            "n_params": blk["n_params"],
            "wires": blk["wires"],
            # committed 5-seed aggregates -- the HEADLINE numbers of record,
            # distinct from the single live seed above (Rule 6: label both).
            "committed_mean_5seed": blk["mean"],
            "committed_sd_5seed": blk["sd"],
            "committed_per_seed": blk["per_seed"],
            "floor_mean_5seed": res["floors"][arm]["mean"],
            "floor_sd_5seed": res["floors"][arm]["sd"],
            "floor_clearance": res["floor_clearance"][arm],
            "comparator_mean_5seed": res["comparator"]["arms"][arm]["mean"],
            "comparator_sd_5seed": res["comparator"]["arms"][arm]["sd"],
            "comparator_feature_dim": res["comparator"]["arms"][arm]["feature_dim"],
            "comparator_minus_circuit":
                res["comparator"]["comparator_minus_circuit"][arm],
            "bound_family": fam,
            "bound_value": pred["grade1_bounds"][fam]["bound_value"],
            "bound_fraction": pred["grade1_bounds"][fam]["fraction"],
            "bound_definition": pred["grade1_bounds"][fam]["definition"],
            "bound_exact": pred["grade1_bounds"][fam]["exact"],
            "purity_contrast": res["purity_contrast_doc55_SS19_4"][arm],
            "per_q_reported": res["reported_not_registered_doc55_SS6_9"][
                "per_arm"][arm]["by_query_index_PRIMACY_RECENCY"],
        }

    # --------------------------------- the frozen per-noun angle table (lookup)
    nonocc = set(INS.CATEGORIES["nonoccupational"]["members"])
    faail = set(n for n in INS.HUMANS if n in WZ.FAAIL)
    nouns = {}
    for nname in INS.HUMANS:
        p1, p2 = fm["coords"][nname]
        nouns[nname] = {
            "gloss": INS.GLOSS[nname],
            "p1": float(p1), "p2": float(p2),
            "theta_pc1": float(fm["theta_pc1"][nname]),
            "theta_pc2": float(fm["theta_pc2"][nname]),
            "phi_atan2": float(fm["phi_exp42"][nname]),
            "theta_wazn": float(fm["theta_wazn"][nname]),
            "is_bearer_nonoccupational": nname in nonocc,
            "is_bearer_faail": nname in faail,
        }

    # -------------------------- the client-side arc calculus, computed in Python
    # so the browser's identical loop can be gated against it.
    arcs = {}
    for key, angkey, member in (("FA_nonoccupational", "phi_atan2", nonocc),
                                ("WZ_wazn_faail", "theta_wazn", faail)):
        names = sorted(nouns)
        a = [nouns[n][angkey] for n in names]
        lab = [n in member for n in names]
        best, n_, win = arc_bound(a, lab)
        arcs[key] = {
            "angle_field": angkey,
            "n_nouns": n_,
            "best_correct": best,
            "best_accuracy": best / n_,
            "window_start_index_in_angle_order": win[0],
            "window_length": win[1],
            "statement": ("EXACT. A d = 3 readout is p(theta) = a + R cos(theta - "
                          "phi), so the YES region is ONE contiguous arc and the "
                          "circle is cut into at most 2 arcs -- the M = 2 row of "
                          "diagnostic_readout_arity_bounds.json's proved "
                          "exact_maximum_2M_minus_2."),
            "scope": ("PER-NOUN category readout over the %d-noun inventory. This "
                      "is NOT the committed per-twin bound (denominator 160 or "
                      "270) and the two must not be compared." % n_),
        }
        print("[export] arc bound %-20s %d/%d = %.4f" % (key, best, n_, best / n_))

    model = {
        "experiment": "exp69",
        "page": "language lab",
        "spec": res["spec"],
        "generated_by": "export_demo_lab.py",
        "arm_order": ARMS,
        "arms": arms_out,
        "verbs": verbs,
        "verb_slots": slots,
        "nouns": nouns,
        "categories": {
            "nonoccupational": {
                "key": "nonoccupational",
                "gloss": INS.CATEGORIES["nonoccupational"]["gloss"],
                "members": sorted(nonocc),
                "flag": "is_bearer_nonoccupational",
                "question_gloss": "the one who [verb] -- is he in a non-occupational state?",
                "source": store["A"][1]["category"]["source"],
                "feature_family": store["A"][1]["category"]["feature_family"],
            },
            "wazn_faail": {
                "key": "wazn_faail",
                "gloss": store["B"][1]["category"]["gloss"],
                "members": sorted(faail),
                "flag": "is_bearer_faail",
                "question_gloss": "the one who [verb] -- is he a faail-pattern word?",
                "source": store["B"][1]["category"]["source"],
                "feature_family": store["B"][1]["category"]["feature_family"],
            },
        },
        "question_form_doc55_SS4_2": {
            "surface_template": "الذي %s هل هو %s",
            "marker": "المميز",
            "marker_note": ("the marker names no referent, so both twins share "
                            "the question surface EXACTLY"),
            "label_rule": "1 iff agent(event q) is a category bearer",
            "query_wires": ("the question effect always acts on wires (0, 1) in "
                            "referent-list order -- identical for both twins; the "
                            "role information lives in the EVENT wiring"),
        },
        "pregroup_typing": {
            "source": "archive/01_parser_pipeline/arabic_dep_reader.py:130-180",
            "SVO_verb_type": "n.r @ s @ n.l",
            "VSO_verb_type": "s @ n.l @ n.l",
            "SVO_reduction": "Cup(n, n.r) @ Id(s) @ Cup(n.l, n)",
            "VSO_swap": "Id(s) @ Id(n.l) @ Swap(n.l, n) @ Id(n)",
            "VSO_reduction": "words >> swap >> (Id(s) @ Cup(n.l, n) @ Cup(n.l, n))",
            "invariance_caveat": res["reported_not_registered_doc55_SS6_9"][
                "surface_order_caveat"],
            "reconciliation": res["meta_notes"][
                "word_order_reconciliation_ORCHESTRATOR_REVIEW"],
        },
        "arc_calculus": arcs,
        "arity_rule": {
            "source": "diagnostic_readout_arity_bounds.json",
            "exact_maximum_arcs": "2M - 2",
            "at_M_2": 2,
            "design_rule": "M >= A/2 + 1 for a category carving A arcs",
            "why_no_M_gt_2_arm": ("doc 55 SS4.6: exp67 measured arity_lifts = "
                                  "FALSE while the classical comparator on the "
                                  "identical features DID convert the same "
                                  "budget. The budget exists and in-circuit "
                                  "conversion failed."),
        },
        "kernel_factorisation": kern,
        "comparator": {
            "arbiter_reading": res["comparator"]["arbiter_reading"],
            "function_class_qualifier": res["comparator"]["function_class_qualifier"],
            "comparator_minus_circuit": res["comparator"]["comparator_minus_circuit"],
            "matches_or_exceeds_all_arms":
                res["verdicts"]["comparator_matches_or_exceeds_all_arms"],
        },
        "composition_cost": res["composition_cost_doc55_SS18_2"],
        "verdicts": res["verdicts"],
        "training": res["training"],
        "bounds_context_only": res["bounds_context_only_doc55_SS12"],
        "bounds_are_context_only_note": pred["bounds_are_context_only"],
        "outcome_rows": res["outcome_rows_doc55_SS9_2"],
        "scope_doc55_SS11": res["meta_notes"]["scope_doc55_SS11"],
        "frozen_feature_constants": {
            "note": ("doc 55 SS4.4, FROZEN and imported through "
                     "instrument_sb_bounds.features. Baked in as CONSTANTS: the "
                     "page looks the semantic angles up and never recomputes a "
                     "PCA. The wazn angle is the one exception and is COMPUTED "
                     "in the browser from the exported featuriser basis."),
            "pca": pred["provenance"]["raw_family"]["pca_fit_on"],
            "pca_fit_n": pred["provenance"]["raw_family"]["pca_fit_n"],
            "sign_pin": pred["provenance"]["raw_family"]["sign_pin"],
            "sign_pin_applied": pred["provenance"]["raw_family"][
                "plane_agreement_with_frozen2d"]["sign_pin_applied"],
            "normalisation": pred["provenance"]["raw_family"]["normalisation"],
            "norm_divisors": pred["provenance"]["raw_family"]["norm_divisors"],
            "var_explained_pc1_plus_pc2": pred["provenance"]["raw_family"][
                "var_explained_pc1_plus_pc2"],
            "raw_dim": pred["provenance"]["raw_family"]["raw_dim"],
            "scaling_name": fm["scaling"]["name"],
            "scaling_formula": fm["scaling"]["formula"],
            "theta_formula": "theta_k(r) = (p_k(r) + 1) * pi   ->  [0, 2pi)",
            "FA": "phi(r) = atan2(PC2(r), PC1(r))  -- the frozen exp42 map",
            "WZ": pred["grade1_bounds"]["R-M"]["definition"],
        },
        "why_semantic_mode_is_a_closed_inventory": (
            "the AraVec-derived angle of a noun is the atan2 of its position in "
            "a PCA plane fitted to 44 raw 300-dimensional word vectors. Those "
            "vectors live in a %d-dimensional model that is NOT shipped with "
            "this page (and whose licence is unresolved), so an arbitrary word "
            "has no computable angle here. The 24 angles below are frozen "
            "constants. The wazn arm has no such restriction because its angle "
            "is computed from the SURFACE FORM."
            % pred["provenance"]["raw_family"]["raw_dim"]),
        "inventory": {
            "n_humans": len(nouns),
            "n_referents_per_story": 2,
            "n_verbs": len(verbs),
            "embedding_dim": pred["provenance"]["raw_family"]["raw_dim"],
            "pca_fit_population": pred["provenance"]["raw_family"]["pca_fit_n"],
        },
        "provenance": {
            "results_json": RESULTS,
            "results_sha256": sha256_file(RESULTS),
            "predictions_json": PREDICTIONS,
            "predictions_sha256": sha256_file(PREDICTIONS),
            "kernel_json": KERNEL,
            "kernel_sha256": sha256_file(KERNEL),
            "dataset_A": R.DATA_A, "dataset_A_sha256": store["A"][2],
            "dataset_B": R.DATA_B, "dataset_B_sha256": store["B"][2],
            "raw_vectors": RAW, "raw_vectors_sha256": sha256_file(RAW),
            "frozen_embeddings": "exp42_noun_embeddings.json",
            "frozen_embeddings_sha256": fm["embeddings_sha256"],
            "bounds_diagnostic": DIAG_SB,
            "bounds_diagnostic_sha256": sha256_file(DIAG_SB),
            "featurizer": "exp65_featurizer.py",
            "featurizer_sha256": sha256_file("exp65_featurizer.py"),
            "comparator_json": res["comparator"]["source"],
            "design_doc": res["spec"],
            "a0_anchor": {
                "bit_exact": res["a0_anchor"]["bit_exact"],
                "max_abs_per_seed_delta": res["a0_anchor"]["max_abs_per_seed_delta"],
                "note": res["a0_anchor"]["component"],
            },
            "note": ("exp69 SB-DisCoCirc rung: 6 verbs in two disjoint slots, 2 "
                     "referents, a 24-noun human inventory, constructed twin "
                     "stories, simulation only. The classical comparator ties "
                     "or exceeds every arm."),
        },
    }
    json.dump(model, open(os.path.join(OUT, "model_exp69.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[export] wrote demo/model_exp69.json (%d arms, %d nouns, %d verbs)"
          % (len(arms_out), len(nouns), len(verbs)))

    # ------------------------------------------------------------ the testsets
    sets = {}
    for part in ("A", "B"):
        items = store[part][0]
        for T in (1, 2):
            test = [i for i in items if i["split"] == "test" and i["T"] == T]
            by_pair = {}
            for it in test:
                by_pair.setdefault(it["pair_id"], []).append(it["id"])
            out_items = []
            for it in test:
                wires = {r["name"]: i for i, r in enumerate(it["referents"])}
                q = it["question"]
                out_items.append({
                    "id": it["id"], "pair_id": it["pair_id"],
                    "twin_ids": sorted(by_pair[it["pair_id"]]),
                    "K": len(wires), "T": it["T"],
                    "query_index": it["query_index"],
                    "queried_verb": it["queried_verb"],
                    "referents": [r["name"] for r in it["referents"]],
                    "intro": [[w, n] for n, w in
                              sorted(wires.items(), key=lambda kv: kv[1])],
                    "events": [[e["verb"], wires[e["agent"]], wires[e["patient"]]]
                               for e in it["events"]],
                    "question": [[q["v"]], wires[q["a"]], wires[q["p"]]],
                    "answer": comp.LABELS[it["answer"]],
                    "answer_label": it["answer"],
                    "sentences": it["sentences"],
                    "order_flags": it["order_flags"],
                    "question_surface": q["surface"],
                })
            sets["%s_T%d" % (part, T)] = {
                "part": part, "T": T,
                "dataset": R.DATA_A if part == "A" else R.DATA_B,
                "dataset_sha256": store[part][2],
                "n_items": len(out_items), "n_pairs": len(by_pair),
                "items": out_items,
            }
            print("[export]   testset %s_T%d: %d items, %d pairs"
                  % (part, T, len(out_items), len(by_pair)))

    ts = {"experiment": "exp69",
          "split": ("test -- held-out unseen PAIRINGS (doc 55 SS4.5); the split "
                    "is over pairings, NOT over nouns, so nothing here tests "
                    "generalisation to words the model never met"),
          "compiled_by": ("export_demo_lab.py, mirroring exp69_run.compile_item "
                          "/ exp42_compiler.compile_story; angles are looked up "
                          "per arm from model_exp69.json nouns[]"),
          "sets": sets}
    json.dump(ts, open(os.path.join(OUT, "testset_lab.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[export] wrote demo/testset_lab.json (%d sets)" % len(sets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
