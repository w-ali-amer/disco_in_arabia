# -*- coding: utf-8 -*-
"""extract_aravec_candidates.py -- D4 (doc 59 SS2/SS7) CANDIDATE extraction.

Produces the RAW MATERIAL for a large Arabic referent inventory. It does NOT
decide membership in anything.

    *** NO MEMBERSHIP, NO HUMAN-NESS, IS CONFIRMED BY THIS SCRIPT ***
    Every noun it emits is a CANDIDATE admitted by a stated morphological
    rule (CAMeL Tools' `rat` rationality feature over the AraVec vocabulary).
    A rule is not a confirmation. Doc 47 SS13.4 / doc 50 SS5.2 discipline
    requires membership to be a user-confirmed design input; doc 59 SS6 poses
    the gate PROTOCOL for hundreds of nouns as its own user decision. Until
    that protocol runs, nothing here may be used as a category, a label, a
    training signal, or a claim that a noun denotes a human.

Four stages, each a separate process so no stage's peak memory is another's
problem (the WSL box runs the experiment pool concurrently):

  vocab      [WSL] mmap the AraVec model, dump the top-N vocabulary tokens
                   (by corpus count) that are surface ال-forms, with counts
                   and frequency ranks, to a scratch JSON.
  candidates [WSL] CAMeL-analyse those tokens, apply the admission rules,
                   write candidate_human_nouns.json.
  vectors    [WSL] mmap the model again, write the raw 300-dim vectors for
                   the candidates to aravec_raw_vectors_candidates.json with
                   a canonical sha256, mirroring aravec_raw_vectors_44.json.
  diagnose   [Mac] write diagnostic_inventory_candidates.json: source-rule
                   counts, PCA-plane coverage against the COMMITTED plane,
                   nearest-neighbour spread, duplicate/variant groups.

PRECEDENT FOLLOWED EXACTLY (aravec_raw_vectors_44.json, extracted 2026-08-23):
  * source model            aravec/full_uni_cbow_300_twitter.mdl
  * ta-marbuta convention   U+0629 -> U+0647 at lookup time, as
                            exp42_make_embeddings.py; noun KEYS keep the
                            conventional orthography (ة), the LOOKUP is
                            normalised.
  * canonical sha256        sha256(json.dumps(vectors, ensure_ascii=False,
                            sort_keys=True).encode("utf-8")), recorded in the
                            artifact's own meta.
  * the `vectors` stage re-extracts the committed 44 as a self-check and
    asserts their canonical sha reproduces 699450d3... -- i.e. this script's
    extraction path is bit-identical to the committed one before it is
    trusted for anything new.

MODEL LOAD NOTE: the model is opened with mmap="r". The WSL box has ~7 GB of
RAM and an experiment pool resident in it; a non-mmap load short-reads the
1.5 GB .npy. mmap changes nothing about the values (same bytes, same float32
-> float64 cast) and the 44-noun sha self-check proves it.

Usage:
    python extract_aravec_candidates.py vocab      --out SCRATCH.json
    python extract_aravec_candidates.py candidates --vocab SCRATCH.json \
                                                   --frozen44 aravec_raw_vectors_44.json \
                                                   --out candidate_human_nouns.json
    python extract_aravec_candidates.py vectors    --candidates candidate_human_nouns.json \
                                                   --frozen44 aravec_raw_vectors_44.json \
                                                   --out aravec_raw_vectors_candidates.json
    python extract_aravec_candidates.py diagnose   --candidates candidate_human_nouns.json \
                                                   --vectors aravec_raw_vectors_candidates.json \
                                                   --out diagnostic_inventory_candidates.json
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import socket
import sys

ARAVEC = "aravec/full_uni_cbow_300_twitter.mdl"

# The committed 44-noun artifact's canonical vectors sha (self-check target).
SHA44 = "699450d3d292f789d368920c33febf49ee4fe4ceff63501308b6da5759d6ce11"

# doc 46 SS5 / doc 47 SS13's 24 HUMAN nouns, copied BY NAME from
# instrument_sb_bounds.py. Used only for (a) exclusion from re-extraction and
# (b) rule-behaviour reporting -- never as a membership authority for anything
# outside those 24.
HUMANS_24 = [
    "الطالب", "الرجل", "الولد", "المدير", "المعلم", "الطبيب",
    "العامل", "المهندس", "الشرطي", "الكاتب", "الطباخ", "الجندي",
    "الفلاح", "الحارس", "اللاعب", "الموظف", "الصحفي", "السائق",
    "المسافر", "الضابط", "الباحث", "القائد", "الصياد", "المريض",
]

# ------------------------------------------------------------------ knobs
# Frequency-rank cap on the vocabulary scan. Recorded in every artifact's meta
# so the population a count is stated over is never ambiguous.
SCAN_CAP = 300000
# Candidate band from doc 59 SS6 ("~500"): fill strictest tier first, stop here.
TARGET_MAX = 600

AL_FORM = re.compile("^ال[ء-ي]{2,}$")   # bare Arabic letters only
TA_MARBUTA, HA = "ة", "ه"

# Surface wazn tags. DESCRIPTIVE metadata attached to the admitting analysis's
# lemma -- they are not admission rules and carry no semantic claim.
def wazn_tag(lex_bare):
    if re.match("^م[ء-ي]{3,}$", lex_bare):
        return "mu_participle"          # مُفعِّل / مُفتَعِل ... (م-initial deverbal)
    if lex_bare.endswith("ي") and len(lex_bare) >= 4:
        return "nisba_yaa"              # نسبة -يّ person nouns
    if re.match("^[ء-ي]ا[ء-ي][ء-ي]$", lex_bare):
        return "faail"                  # فاعل agent noun
    if re.match("^[ء-ي][ء-ي]ا[ء-ي]$", lex_bare):
        return "faaal_or_fa33aal"       # فعّال / فعال intensive-agent
    if re.match("^[ء-ي][ء-ي]ي[ء-ي]$", lex_bare):
        return "faiil"                  # فعيل
    return "other"


def strip_diac(s):
    return re.sub("[ً-ْـٰ]", "", s)


def norm_ta_marbuta(w):
    return w.replace(TA_MARBUTA, HA)


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def stamp():
    return "%s %s" % (socket.gethostname(),
                      datetime.date.today().isoformat())


NO_MEMBERSHIP = (
    "CANDIDATES ONLY. No membership, category, or human-ness is confirmed "
    "here. Admission is by the stated morphological rule (CAMeL Tools `rat` "
    "over the AraVec vocabulary), which is evidence, not confirmation. The "
    "doc 59 SS6 gate protocol is a separate user decision and has NOT run.")


# =========================================================== stage: vocab
def stage_vocab(args):
    import numpy as np
    from gensim.models import Word2Vec

    wv = Word2Vec.load(args.model, mmap="r").wv
    keys = list(wv.key_to_index)
    counts = np.asarray(wv.expandos["count"])
    assert len(keys) == len(counts)
    # gensim's key order is NOT frequency order for this model -- sort here,
    # ties broken by the model's own key order so the scan is deterministic.
    order = sorted(range(len(keys)), key=lambda i: (-int(counts[i]), i))
    cap = min(args.cap, len(order))
    rows = []
    for rank in range(cap):
        i = order[rank]
        k = keys[i]
        if AL_FORM.match(k):
            rows.append({"token": k, "count": int(counts[i]), "rank": rank})
    out = {
        "meta": {
            "stage": "vocab",
            "model": args.model,
            "vocab_size": len(keys),
            "scan_cap_by_frequency_rank": cap,
            "min_count_at_cap": int(counts[order[cap - 1]]),
            "al_form_regex": AL_FORM.pattern,
            "n_al_forms": len(rows),
            "extracted_on": stamp(),
        },
        "tokens": rows,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[vocab] vocab=%d scanned=%d al_forms=%d -> %s"
          % (len(keys), cap, len(rows), args.out))


# ====================================================== stage: candidates
def _readings(an, token):
    """All CAMeL analyses of `token` that are genuinely ال-definite."""
    out = []
    for a in an.analyze(token):
        prc0 = a.get("prc0")
        definite = (prc0 == "Al_det") or str(a.get("gloss", "")).startswith("the+")
        if definite:
            out.append(a)
    return out


def _classify(readings):
    """Apply the admission rules to one token's definite readings.

    Returns (tier, admitting_analysis, stats) or (None, None, stats).

    Rules, strictest first. DN = the definite NOUN readings (pos == 'noun';
    noun_prop and adj are excluded). Distinctness is by lemma+rationality, so
    a lemma listed once per case ending does not inflate a tally.

      camel_rat_r_unanimous  every distinct DN lemma is rat == 'r'
      camel_rat_r_majority   > 1/2 of distinct DN lemmas are rat == 'r'
      camel_rat_r_any        at least one DN lemma is rat == 'r'

    All three additionally REQUIRE a masculine-singular rational reading
    (gen == 'm', num == 's'): the committed inventory's 24 HUMANS are all
    definite masculine singular, and matching that convention is what makes a
    candidate droppable into the existing referent slot without a new
    agreement story. This is a CONVENTION filter, not a semantic one, and it
    excludes feminine and plural human-denoting forms by construction.
    """
    dn = [a for a in readings if a.get("pos") == "noun"]
    lex_r, lex_all = set(), set()
    admit = None
    for a in dn:
        # DIACRITIZED lemma is the distinctness key. Undiacritized keys would
        # merge vocalization homographs -- رِجْل 'leg' (rat=i) with رَجُل 'man'
        # (rat=r) -- and silently promote every such token to "unanimous",
        # inflating exactly the ambiguity class the tiers exist to grade.
        lex_all.add(str(a.get("lex", "")))
        if a.get("rat") == "r":
            lex_r.add(str(a.get("lex", "")))
            if a.get("gen") == "m" and a.get("num") == "s" and admit is None:
                admit = a
    stats = {"n_definite_readings": len(readings), "n_noun_readings": len(dn),
             "n_distinct_lemmas": len(lex_all), "n_distinct_rational_lemmas": len(lex_r)}
    if admit is None or not lex_all:
        return None, None, stats
    frac = len(lex_r) / float(len(lex_all))
    if len(lex_r) == len(lex_all):
        tier = "camel_rat_r_unanimous"
    elif frac > 0.5:
        tier = "camel_rat_r_majority"
    else:
        tier = "camel_rat_r_any"
    stats["rational_lemma_fraction"] = frac
    return tier, admit, stats


TIERS = ["camel_rat_r_unanimous", "camel_rat_r_majority", "camel_rat_r_any"]


def stage_candidates(args):
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer

    voc = json.load(open(args.vocab, encoding="utf-8"))
    tokens = voc["tokens"]
    frozen44 = json.load(open(args.frozen44, encoding="utf-8"))["vectors"]
    excl = set(frozen44) | set(norm_ta_marbuta(n) for n in frozen44)

    db = MorphologyDB.builtin_db()
    an = Analyzer(db, cache_size=200000)

    funnel = {"scanned_vocab_tokens": voc["meta"]["scan_cap_by_frequency_rank"],
              "al_form_tokens": len(tokens),
              "already_in_frozen_44": 0,
              "no_definite_analysis": 0,
              "no_definite_noun_reading": 0,
              "no_rational_reading": 0,
              "rational_but_not_masc_sing": 0,
              "admitted_before_dedup": 0,
              "dropped_as_orthographic_duplicate": 0,
              "admitted_after_dedup": 0,
              "kept_after_band_cap": 0}
    per_tier_total = dict((t, 0) for t in TIERS)

    admitted = {}          # noun key -> record
    for i, row in enumerate(tokens):
        tok = row["token"]
        if (i + 1) % 5000 == 0:
            print("[candidates] %d/%d analysed, admitted %d"
                  % (i + 1, len(tokens), len(admitted)), flush=True)
        if tok in excl:
            funnel["already_in_frozen_44"] += 1
            continue
        rd = _readings(an, tok)
        if not rd:
            funnel["no_definite_analysis"] += 1
            continue
        tier, admit, st = _classify(rd)
        if tier is None:
            dn = [a for a in rd if a.get("pos") == "noun"]
            if not dn:
                funnel["no_definite_noun_reading"] += 1
            elif st["n_distinct_rational_lemmas"] == 0:
                funnel["no_rational_reading"] += 1
            else:
                funnel["rational_but_not_masc_sing"] += 1
            continue
        funnel["admitted_before_dedup"] += 1
        per_tier_total[tier] += 1

        lex = str(admit.get("lex", ""))
        lex_bare = strip_diac(lex)
        # Orthographic key: restore ta-marbuta iff the admitting lemma carries
        # one (AraVec's vocabulary stores ة as ه; the committed 44 keep ة in
        # the KEY and normalise only at lookup).
        noun = tok
        if lex_bare.endswith(TA_MARBUTA) and tok.endswith(HA):
            noun = tok[:-1] + TA_MARBUTA
        rec = {
            "noun": noun,
            "aravec_key": tok,
            "provenance": tier,
            "provenance_detail": {
                "rule": ("admitted because CAMeL Tools gives this token a "
                         "definite (Al_det) masculine-singular NOUN reading "
                         "with rationality rat='r'; tier records how much of "
                         "the token's noun reading set is rational"),
                "rational_lemma_fraction": st.get("rational_lemma_fraction"),
                "n_definite_readings": st["n_definite_readings"],
                "n_noun_readings": st["n_noun_readings"],
                "n_distinct_lemmas": st["n_distinct_lemmas"],
                "n_distinct_rational_lemmas": st["n_distinct_rational_lemmas"],
            },
            "lemma": lex,
            "lemma_bare": lex_bare,
            "root": admit.get("root"),
            "pattern": admit.get("pattern"),
            "wazn_tag": wazn_tag(lex_bare),
            "camel_stemgloss": admit.get("stemgloss"),
            "aravec_count": row["count"],
            "aravec_freq_rank": row["rank"],
            "confirmed_human": None,
            "confirmation_status": "UNCONFIRMED -- doc 59 SS6 gate protocol has not run",
        }
        prev = admitted.get(noun)
        if prev is None:
            admitted[noun] = rec
        else:
            funnel["dropped_as_orthographic_duplicate"] += 1
            if row["count"] > prev["aravec_count"]:
                admitted[noun] = rec
    funnel["admitted_after_dedup"] = len(admitted)

    # Band fill: strictest tier first, then by corpus frequency. Deterministic.
    ranked = sorted(admitted.values(),
                    key=lambda r: (TIERS.index(r["provenance"]),
                                   -r["aravec_count"], r["noun"]))
    kept = ranked[:args.target_max]
    funnel["kept_after_band_cap"] = len(kept)

    # ---- rule behaviour on the committed 44 (NOT a membership statement) ----
    rule_on_44 = {"humans_24_readmitted": [], "humans_24_rejected": [],
                  "objects_20_admitted_by_rule": [], "objects_20_rejected": []}
    objects_20 = [n for n in frozen44 if n not in set(HUMANS_24)]
    for n in list(HUMANS_24) + objects_20:
        rd = _readings(an, n)
        tier, admit, _ = _classify(rd)
        if n in set(HUMANS_24):
            (rule_on_44["humans_24_readmitted"] if tier
             else rule_on_44["humans_24_rejected"]).append(
                {"noun": n, "tier": tier})
        else:
            (rule_on_44["objects_20_admitted_by_rule"] if tier
             else rule_on_44["objects_20_rejected"]).append(
                {"noun": n, "tier": tier})

    tier_counts_kept = {}
    wazn_counts = {}
    for r in kept:
        tier_counts_kept[r["provenance"]] = tier_counts_kept.get(r["provenance"], 0) + 1
        wazn_counts[r["wazn_tag"]] = wazn_counts.get(r["wazn_tag"], 0) + 1

    art = {
        "meta": {
            "purpose": ("D4 (doc 59 SS2/SS7) CANDIDATE inventory of Arabic "
                        "definite masculine-singular nouns with a rational "
                        "(human-denoting) morphological reading; raw material "
                        "for a large referent inventory"),
            "HEADER_WARNING_NO_MEMBERSHIP": NO_MEMBERSHIP,
            "convention": ("definite ال-form, masculine singular, matching the "
                           "committed 24 HUMANS of instrument_sb_bounds.py; "
                           "ta-marbuta U+0629 -> U+0647 at AraVec lookup, key "
                           "keeps ة (as aravec_raw_vectors_44.json)"),
            "convention_exclusions": ("feminine and plural human-denoting forms "
                                      "are excluded BY CONVENTION, not by "
                                      "semantics; see funnel."),
            "sources_considered": {
                "camel_tools": "USED -- builtin morphology DB, `rat` feature",
                "aravec_vocabulary": ("USED -- candidate universe is the AraVec "
                                      "vocabulary itself, so coverage is true "
                                      "by construction, not hoped for"),
            },
            "admission_rules": {
                "camel_rat_r_unanimous": "every distinct definite-noun lemma of the token is rat='r'",
                "camel_rat_r_majority": "more than half of distinct definite-noun lemmas are rat='r'",
                "camel_rat_r_any": "at least one distinct definite-noun lemma is rat='r'",
                "shared_requirement": "a definite masculine-singular rat='r' NOUN reading must exist",
                "excluded_pos": "noun_prop and adj readings never admit",
            },
            "aravec_model": args.model,
            "camel_db": "camel_tools MorphologyDB.builtin_db()",
            "scan_cap_by_frequency_rank": voc["meta"]["scan_cap_by_frequency_rank"],
            "min_count_at_cap": voc["meta"]["min_count_at_cap"],
            "band_cap": args.target_max,
            "n_candidates": len(kept),
            "counts_by_provenance_kept": tier_counts_kept,
            "counts_by_provenance_all_admitted": per_tier_total,
            "counts_by_wazn_tag_kept": wazn_counts,
            "funnel": funnel,
            "aravec_coverage": ("100% by construction -- every candidate's "
                                "aravec_key is a vocabulary token of the model"),
            "frozen_44_excluded_from_extraction": sorted(frozen44),
            "rule_behaviour_on_committed_44": rule_on_44,
            "extracted_on": stamp(),
            "generator": os.path.basename(os.path.abspath(__file__)),
            "generator_sha256": file_sha256(os.path.abspath(__file__)),
        },
        "candidates": kept,
    }
    art["meta"]["candidates_sha256"] = canonical_sha256(kept)
    json.dump(art, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[candidates] kept %d (%s) -> %s"
          % (len(kept), tier_counts_kept, args.out))


# ========================================================= stage: vectors
def _lookup(wv, noun):
    if noun in wv.key_to_index:
        return noun
    k = norm_ta_marbuta(noun)
    if k in wv.key_to_index:
        return k
    raise KeyError("noun %r not in AraVec even after ta-marbuta normalization" % noun)


def stage_vectors(args):
    import numpy as np
    from gensim.models import Word2Vec

    cand = json.load(open(args.candidates, encoding="utf-8"))
    nouns = [c["noun"] for c in cand["candidates"]]
    wv = Word2Vec.load(args.model, mmap="r").wv

    # --- self-check: reproduce the committed 44-noun artifact bit for bit ---
    f44 = json.load(open(args.frozen44, encoding="utf-8"))
    re44 = {}
    for n in f44["vectors"]:
        re44[n] = [float(x) for x in np.asarray(wv[_lookup(wv, n)], dtype=np.float64)]
    sha_re44 = canonical_sha256(re44)
    ok44 = (sha_re44 == f44["meta"]["vectors_sha256"] == SHA44)
    print("[vectors] 44-noun self-check sha=%s ok=%s" % (sha_re44, ok44))
    if not ok44:
        sys.exit("ABORT: extraction path does not reproduce the committed 44 vectors")

    vectors, keys_used = {}, {}
    for n in nouns:
        k = _lookup(wv, n)
        keys_used[n] = k
        vectors[n] = [float(x) for x in np.asarray(wv[k], dtype=np.float64)]
    assert len(vectors) == len(nouns), "duplicate noun keys in candidate list"

    art = {
        "meta": {
            "purpose": ("raw 300-dim AraVec vectors, %d CANDIDATE nouns, for "
                        "the D4 big-inventory direction (doc 59 SS2)" % len(vectors)),
            "HEADER_WARNING_NO_MEMBERSHIP": NO_MEMBERSHIP,
            "source": args.model,
            "convention": "ta-marbuta U+0629 -> U+0647, as exp42_make_embeddings.py",
            "extracted_on": "WSL " + stamp(),
            "n_nouns": len(vectors),
            "candidate_list": os.path.basename(args.candidates),
            "candidate_list_sha256": file_sha256(args.candidates),
            "declared_candidates_sha256": cand["meta"]["candidates_sha256"],
            "excludes": ("the 44 nouns of aravec_raw_vectors_44.json -- already "
                         "committed, not re-extracted"),
            "precedent_44_reextracted_sha256": sha_re44,
            "precedent_44_sha_reproduced": bool(ok44),
            "model_loaded_with_mmap": True,
            "lookup_keys_differing_from_noun": {n: k for n, k in keys_used.items()
                                                if k != n},
            "generator": os.path.basename(os.path.abspath(__file__)),
            "generator_sha256": file_sha256(os.path.abspath(__file__)),
        },
        "vectors": vectors,
    }
    art["meta"]["vectors_sha256"] = canonical_sha256(vectors)
    json.dump(art, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[vectors] wrote %s  n=%d  vectors_sha256=%s"
          % (args.out, len(vectors), art["meta"]["vectors_sha256"]))


# ======================================================== stage: diagnose
# The committed plane's registered constants, from diagnostic_sb_bounds.json
# (families.raw). Reproduced, then ASSERTED -- never assumed.
COMMITTED = {
    "norm_divisors": [8.669494516224397, 10.160404144924657],
    "var_explained_pc1": 0.12685513291325928,
    "var_explained_pc2": 0.08050060927190729,
    "sign_pin_applied": [-1, 1],
}


def _hull(points):
    """Monotone-chain convex hull; returns hull vertices CCW."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    hi = []
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _inside(poly, p):
    """Point-in-convex-polygon (CCW), boundary counts as inside."""
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
            return False
    return True


def stage_diagnose(args):
    import numpy as np

    here = os.path.dirname(os.path.abspath(args.out)) or "."
    f44_path = os.path.join(here, "aravec_raw_vectors_44.json")
    frz_path = os.path.join(here, "exp42_noun_embeddings.json")
    cand = json.load(open(args.candidates, encoding="utf-8"))
    cvec = json.load(open(args.vectors, encoding="utf-8"))
    f44 = json.load(open(f44_path, encoding="utf-8"))

    # --- sha re-verification, on the Mac, of both transferred artifacts ---
    sha_checks = {
        "candidates_recomputed": canonical_sha256(cand["candidates"]),
        "candidates_declared": cand["meta"]["candidates_sha256"],
        "vectors_recomputed": canonical_sha256(cvec["vectors"]),
        "vectors_declared": cvec["meta"]["vectors_sha256"],
        "frozen44_recomputed": canonical_sha256(f44["vectors"]),
        "frozen44_declared": f44["meta"]["vectors_sha256"],
    }
    sha_checks["all_match"] = bool(
        sha_checks["candidates_recomputed"] == sha_checks["candidates_declared"]
        and sha_checks["vectors_recomputed"] == sha_checks["vectors_declared"]
        and sha_checks["frozen44_recomputed"] == sha_checks["frozen44_declared"] == SHA44)
    assert sha_checks["all_match"], "sha re-verification FAILED: %r" % sha_checks

    # ---------------- rebuild the COMMITTED plane (load_raw's recipe) -------
    v44 = f44["vectors"]
    names44 = sorted(v44)                      # pca_fit_on = "all44"
    X = np.array([v44[n] for n in names44], dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum())
    Pall = Xc @ Vt[:2].T
    idx = {n: i for i, n in enumerate(names44)}
    P24 = np.array([Pall[idx[n]] for n in HUMANS_24])
    fz = json.load(open(frz_path, encoding="utf-8"))["embeddings"]
    F = np.array([fz[n]["vec2"] for n in HUMANS_24], dtype=float)
    U2 = P24 / np.linalg.norm(P24, axis=1, keepdims=True)
    dmin, best = None, None
    for sa in (1, -1):
        for sb in (1, -1):
            dd = float(np.abs(U2 * np.array([sa, sb]) - F).max())
            if dmin is None or dd < dmin:
                dmin, best = dd, (sa, sb)
    P24 = P24 * np.array(best, dtype=float)
    denom = np.abs(P24).max(axis=0)

    plane_check = {
        "recomputed_norm_divisors": [float(denom[0]), float(denom[1])],
        "committed_norm_divisors": COMMITTED["norm_divisors"],
        "recomputed_var_explained": [float(var[0]), float(var[1])],
        "committed_var_explained": [COMMITTED["var_explained_pc1"],
                                    COMMITTED["var_explained_pc2"]],
        "recomputed_sign_pin": list(best),
        "committed_sign_pin": COMMITTED["sign_pin_applied"],
        "max_abs_coord_diff_vs_frozen_vec2": dmin,
    }
    plane_check["verified"] = bool(
        abs(denom[0] - COMMITTED["norm_divisors"][0]) < 1e-9
        and abs(denom[1] - COMMITTED["norm_divisors"][1]) < 1e-9
        and abs(var[0] - COMMITTED["var_explained_pc1"]) < 1e-9
        and abs(var[1] - COMMITTED["var_explained_pc2"]) < 1e-9
        and list(best) == COMMITTED["sign_pin_applied"])
    assert plane_check["verified"], (
        "the rebuilt plane is NOT the committed plane: %r" % plane_check)

    sgn = np.array(best, dtype=float)
    P24n = P24 / denom                          # the 24, registered coords

    names_c = [c["noun"] for c in cand["candidates"]]
    C = np.array([cvec["vectors"][n] for n in names_c], dtype=float)
    PC = ((C - mu) @ Vt[:2].T) * sgn / denom    # candidates on the SAME plane

    # ---------------------------- plane coverage ---------------------------
    hull24 = _hull(P24n.tolist())
    hullC = _hull(PC.tolist())
    inside = sum(1 for p in PC.tolist() if _inside(hull24, p))
    ang24 = np.arctan2(P24n[:, 1], P24n[:, 0]) % (2 * math.pi)
    angC = np.arctan2(PC[:, 1], PC[:, 0]) % (2 * math.pi)
    NB = 36
    b24 = set((ang24 / (2 * math.pi) * NB).astype(int).tolist())
    bC = set((angC / (2 * math.pi) * NB).astype(int).tolist())
    r24 = np.linalg.norm(P24n, axis=1)
    rC = np.linalg.norm(PC, axis=1)

    def _stats(a):
        a = np.asarray(a, dtype=float)
        return {"min": float(a.min()), "p25": float(np.percentile(a, 25)),
                "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
                "max": float(a.max()), "mean": float(a.mean()),
                "sd": float(a.std(ddof=0))}

    plane = {
        "coordinates": ("PC1/PC2 of the COMMITTED plane: PCA fitted on the 44 "
                        "committed vectors only, sign-pinned to exp42's frozen "
                        "vec2, divided by the registered per-component divisors "
                        "max_{24 HUMANS}|PC_k|. Candidates are PROJECTED, never "
                        "refitted, so |coord| > 1 is expected and meaningful."),
        "bbox_24": {"pc1": [float(P24n[:, 0].min()), float(P24n[:, 0].max())],
                    "pc2": [float(P24n[:, 1].min()), float(P24n[:, 1].max())]},
        "bbox_candidates": {"pc1": [float(PC[:, 0].min()), float(PC[:, 0].max())],
                            "pc2": [float(PC[:, 1].min()), float(PC[:, 1].max())]},
        "hull_area_24": _area(hull24),
        "hull_area_candidates": _area(hullC),
        "hull_area_ratio_candidates_over_24": _area(hullC) / _area(hull24),
        "n_candidates_inside_24_hull": int(inside),
        "frac_candidates_inside_24_hull": inside / float(len(names_c)),
        "angular_bins_of_%d_occupied_24" % NB: len(b24),
        "angular_bins_of_%d_occupied_candidates" % NB: len(bC),
        "angular_bins_occupied_by_candidates_not_by_24": len(bC - b24),
        "angular_bins_occupied_by_24_not_by_candidates": len(b24 - bC),
        "radius_24": _stats(r24),
        "radius_candidates": _stats(rC),
        "var_explained_pc1_plus_pc2": float(var[:2].sum()),
        "note": ("plane-FILLING is a geometric statement about where the raw "
                 "vectors land. It says nothing about what any noun denotes."),
    }

    # ------------------------ nearest-neighbour spread ---------------------
    def nn_stats(A, B=None, exclude_self=False):
        A = np.asarray(A, dtype=float)
        B = A if B is None else np.asarray(B, dtype=float)
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        return _stats(d.min(axis=1))

    def cos_nn(A, B=None, exclude_self=False):
        A = np.asarray(A, dtype=float)
        An = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bn = An if B is None else (np.asarray(B, dtype=float)
                                   / np.linalg.norm(B, axis=1, keepdims=True))
        s = An @ Bn.T
        if exclude_self:
            np.fill_diagonal(s, -np.inf)
        return _stats(1.0 - s.max(axis=1))

    X24 = np.array([v44[n] for n in HUMANS_24], dtype=float)
    nn = {
        "raw300_cosine_distance": {
            "candidate_to_nearest_candidate": cos_nn(C, exclude_self=True),
            "human24_to_nearest_human24": cos_nn(X24, exclude_self=True),
            "candidate_to_nearest_of_24": cos_nn(C, X24),
        },
        "plane2d_euclidean_registered_units": {
            "candidate_to_nearest_candidate": nn_stats(PC, exclude_self=True),
            "human24_to_nearest_human24": nn_stats(P24n, exclude_self=True),
            "candidate_to_nearest_of_24": nn_stats(PC, P24n),
        },
        "reading": ("smaller candidate-to-candidate NN than 24-to-24 means the "
                    "candidate cloud is DENSER, not that it is more coherent."),
    }

    # ------------------------ duplicates / variants ------------------------
    by_lemma, by_root, by_tamarbuta = {}, {}, {}
    for c in cand["candidates"]:
        by_lemma.setdefault(c["lemma_bare"], []).append(c["noun"])
        if c.get("root"):
            by_root.setdefault(c["root"], []).append(c["noun"])
        by_tamarbuta.setdefault(norm_ta_marbuta(c["noun"]), []).append(c["noun"])
    dup_lemma = {k: v for k, v in by_lemma.items() if len(v) > 1}
    dup_root = {k: v for k, v in by_root.items() if len(v) > 1}
    dup_ta = {k: v for k, v in by_tamarbuta.items() if len(v) > 1}
    # exact vector collisions (would mean two keys hit one AraVec token)
    seen, vec_dups = {}, {}
    for n in names_c:
        h = hashlib.sha256(json.dumps(cvec["vectors"][n]).encode()).hexdigest()
        if h in seen:
            vec_dups.setdefault(h, [seen[h]]).append(n)
        else:
            seen[h] = n
    variants = {
        "distinct_lemmas": len(by_lemma),
        "distinct_roots": len(by_root),
        "lemma_groups_with_multiple_candidates": len(dup_lemma),
        "root_groups_with_multiple_candidates": len(dup_root),
        "largest_root_groups": sorted(dup_root.items(),
                                      key=lambda kv: -len(kv[1]))[:15],
        "largest_lemma_groups": sorted(dup_lemma.items(),
                                       key=lambda kv: -len(kv[1]))[:15],
        "ta_marbuta_collisions": dup_ta,
        "identical_vector_groups": list(vec_dups.values()),
        "note": ("a shared root is a morphological relation, not a semantic "
                 "one; these groups are flagged for the gate protocol to look "
                 "at, not pruned here."),
    }

    # ------------------------------ source counts --------------------------
    by_tier, by_wazn, by_rank_decile = {}, {}, {}
    for c in cand["candidates"]:
        by_tier[c["provenance"]] = by_tier.get(c["provenance"], 0) + 1
        by_wazn[c["wazn_tag"]] = by_wazn.get(c["wazn_tag"], 0) + 1
        d = min(9, c["aravec_freq_rank"] * 10 // max(1, cand["meta"]["scan_cap_by_frequency_rank"]))
        by_rank_decile["decile_%d" % d] = by_rank_decile.get("decile_%d" % d, 0) + 1

    out = {
        "meta": {
            "purpose": ("D4 candidate-inventory diagnostic: how many candidates "
                        "each source rule produced, how much of the committed "
                        "PC1/PC2 plane they cover, how spread out they are, and "
                        "which are morphological variants of each other"),
            "HEADER_WARNING_NO_MEMBERSHIP": NO_MEMBERSHIP,
            "candidate_file": os.path.basename(args.candidates),
            "candidate_file_sha256": file_sha256(args.candidates),
            "vectors_file": os.path.basename(args.vectors),
            "vectors_file_sha256": file_sha256(args.vectors),
            "frozen44_file_sha256": file_sha256(f44_path),
            "n_candidates": len(names_c),
            "generated_on": stamp(),
            "generator": os.path.basename(os.path.abspath(__file__)),
            "generator_sha256": file_sha256(os.path.abspath(__file__)),
        },
        "sha_reverification_on_mac": sha_checks,
        "committed_plane_check": plane_check,
        "counts": {
            "by_provenance_rule": by_tier,
            "by_provenance_rule_all_admitted_before_band_cap":
                cand["meta"]["counts_by_provenance_all_admitted"],
            "by_wazn_tag": by_wazn,
            "by_frequency_rank_decile": by_rank_decile,
            "funnel": cand["meta"]["funnel"],
        },
        "rule_behaviour_on_committed_44": cand["meta"]["rule_behaviour_on_committed_44"],
        "aravec_coverage": {
            "candidates_in_aravec_vocab": len(names_c),
            "candidates_total": len(names_c),
            "fraction": 1.0,
            "basis": ("candidates were ENUMERATED FROM the AraVec vocabulary and "
                      "every one was re-looked-up during extraction; a missing "
                      "key would have raised, not been skipped"),
            "frozen_44_coverage": "44/44 (committed artifact, unchanged)",
        },
        "plane_coverage": plane,
        "nearest_neighbour": nn,
        "duplicates_and_variants": variants,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[diagnose] wrote %s" % args.out)
    print("[diagnose] plane verified=%s  inside-24-hull=%.3f  hull ratio=%.2f"
          % (plane_check["verified"], plane["frac_candidates_inside_24_hull"],
             plane["hull_area_ratio_candidates_over_24"]))


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("vocab")
    p.add_argument("--model", default=ARAVEC)
    p.add_argument("--cap", type=int, default=SCAN_CAP)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=stage_vocab)

    p = sub.add_parser("candidates")
    p.add_argument("--vocab", required=True)
    p.add_argument("--frozen44", default="aravec_raw_vectors_44.json")
    p.add_argument("--model", default=ARAVEC)
    p.add_argument("--target-max", type=int, default=TARGET_MAX)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=stage_candidates)

    p = sub.add_parser("vectors")
    p.add_argument("--candidates", required=True)
    p.add_argument("--frozen44", default="aravec_raw_vectors_44.json")
    p.add_argument("--model", default=ARAVEC)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=stage_vectors)

    p = sub.add_parser("diagnose")
    p.add_argument("--candidates", required=True)
    p.add_argument("--vectors", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=stage_diagnose)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
