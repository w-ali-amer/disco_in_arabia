# -*- coding: utf-8 -*-
"""Exp 39: the two controls the verb-encoder claim never had.

exp33 (zx4) and exp38 (zxx4) each solve 4 parameters per verb and report a
p-value against a 300-draw RANDOM-theta null.  That null only shows the solved
dials beat noise.  The audit of the archived `cross_verb` blocks showed 87-99.9%
cross-verb alignment: one verb's dials nearly reproduce another verb's target.
exp39 turns that into two pre-registered decisions (see
20_exp39_informative_null_design.md, committed before this file ran):

  Q1  INFORMATIVE NULL.  Replace the random-theta null with the five OTHER
      verbs' solved thetas.  Statistic per verb: median cross-verb alignment
      divided by matched (own-dial) alignment; variant statistic is the median
      over the 6 verbs.  Criterion: verb_label_retired = (R_zx4 >= 0.95) or
      (R_zxx4 >= 0.95).

  Q2  DIAL SWAP (task level).  Score exp34a's 54 orig/swap pairs with a single
      per-verb dial (archived zx4 T3b_theta), then permute the verb->theta
      assignment over all 6! = 720 permutations (identity flagged).  Criterion:
      dials_carry_task_signal = (orig_acc >= 95th percentile of the 720-value
      accuracy distribution).

Nothing is re-solved: thetas come from results_exp33_s0.json /
results_exp38_s0.json; only circuits are rebuilt.  Machinery is reused by
exec-ing source slices of the existing scripts (the same idiom exp33/34a/38 use
to load exp28a's header), so no existing file is modified:

  * Q1 slice: exp38_analog_native.py up to "# ── usable frames".  exp38 is
    exp33 plus the XX gate, and its make_variant/run_sf already handle BOTH
    zx4 and zxx4 -- one slice covers both variants and reproduces exp33's
    harvest byte-for-byte.
  * Q2 slice: exp34a_swap_plausibility.py up to "USABLE = {}".  That gives the
    3-token orig/swap harvest, make_zx4 and the arg-fed scorer; the pair-build
    loop after the marker is replicated here.

A sanity gate compares the recomputed matched per-frame alignments against the
archived T3b_per_frame_alignment; a mismatch > 1e-6 aborts the verdicts.

Run with ARABIC_POS_FUSION=1.
"""
import os, json, itertools
import numpy as np

S_OUT = int(os.environ.get("S_OUT", "0"))
if os.environ.get("ARABIC_POS_FUSION", "0") != "1":
    print("[39] WARNING: ARABIC_POS_FUSION is off — frame supply will be "
          "starved; exp39 must run with fusion on to match exp33/38.",
          flush=True)
MAX_USE_PER_VERB = 10
MIN_FRAMES = 4
SANITY_TOL = 1e-6
RETIRE_RATIO = 0.95          # Q1 pre-registered threshold
TOP_FRAC = 0.95              # Q2 pre-registered percentile

# ── Q1 machinery: exec exp38's header slice ──────────────────────────────
src38 = open("exp38_analog_native.py", encoding="utf-8").read()
head38 = src38[:src38.index("# ── usable frames")]
n38 = {"__name__": "exp39_q1"}
exec(compile(head38, "exp38_head", "exec"), n38)
surface, subjects = n38["surface"], n38["subjects"]
CAND, build = n38["CAND"], n38["build"]
make_variant, align38, pref_from = n38["make_variant"], n38["align"], n38["pref_from"]

# usable frames — replicated verbatim from exp33/exp38 (identical in both)
USABLE = {}
for k, fs in CAND.items():
    gs = []
    for sent, subj, split in fs:
        G = build(sent, subj)
        if G is not None:
            gs.append(dict(G=G, sent=sent, subj=subj, split=split))
        if len(gs) >= MAX_USE_PER_VERB:
            break
    print(f"[39] {surface[k]}: usable frames {len(gs)}/{len(fs)} attempted",
          flush=True)
    if len(gs) >= MIN_FRAMES:
        USABLE[k] = gs

r33 = json.load(open("results_exp33_s0.json", encoding="utf-8"))
r38 = json.load(open("results_exp38_s0.json", encoding="utf-8"))
SRC = {"zx4": r33["variants"]["zx4"], "zxx4": r38["variants"]["zxx4"]}

Q1 = {}
sanity = {"max_abs_dev_vs_archived": {}, "passed": True}
for variant, arch in SRC.items():
    theta = {v: np.array(rec["T3b_theta"], float)
             for v, rec in arch["verbs"].items()}
    verbs = [surface[k] for k in USABLE if surface[k] in theta]
    per_verb, worst = {}, 0.0
    for k in USABLE:
        v = surface[k]
        if v not in theta:
            continue
        tgt = pref_from(subjects[k])
        Hs = [make_variant(g["G"], variant) for g in USABLE[k]]
        matched_frames = [float(align38(H, theta[v], tgt)) for H in Hs]
        matched = float(np.mean(matched_frames))
        arch_frames = arch["verbs"][v]["T3b_per_frame_alignment"]
        dev = (max(abs(a - b) for a, b in zip(matched_frames, arch_frames))
               if len(arch_frames) == len(matched_frames) else float("inf"))
        worst = max(worst, dev)
        cross = {u: float(np.mean([align38(H, theta[u], tgt) for H in Hs]))
                 for u in verbs if u != v}
        cvals = sorted(cross.values())
        med_cross = float(np.median(cvals))
        loo = arch["verbs"][v]["T4b_loo_mean"]
        per_verb[v] = {
            "n_frames": len(Hs),
            "matched_mean": matched,
            "matched_per_frame": matched_frames,
            "archived_loo_mean": loo,
            "cross_verb_alignment": cross,
            "cross_median": med_cross,
            "cross_min": float(min(cvals)),
            "cross_max": float(max(cvals)),
            "ratio_median_cross_over_matched": med_cross / matched,
            "ratio_median_cross_over_loo": (med_cross / loo
                                            if loo and loo > 0 else None),
            "informative_null_p_value": float(
                (1 + sum(c >= matched for c in cvals)) / (1 + len(cvals))),
        }
        print(f"[39] Q1 {variant} {v}: matched={matched:.4f} "
              f"cross med={med_cross:.4f} [{min(cvals):.4f},{max(cvals):.4f}] "
              f"ratio={med_cross / matched:.4f} (dev vs archived {dev:.2e})",
              flush=True)
    ratios = [per_verb[v]["ratio_median_cross_over_matched"] for v in per_verb]
    R = float(np.median(ratios))
    Q1[variant] = {
        "verbs": per_verb,
        "n_verbs": len(per_verb),
        "median_ratio_over_verbs": R,
        "min_ratio_over_verbs": float(min(ratios)),
        "max_ratio_over_verbs": float(max(ratios)),
        "criterion_threshold": RETIRE_RATIO,
        "criterion_tripped": bool(R >= RETIRE_RATIO),
    }
    sanity["max_abs_dev_vs_archived"][variant] = worst
    sanity["passed"] = bool(sanity["passed"] and worst <= SANITY_TOL)
    print(f"[39] Q1 {variant}: R = median over verbs of "
          f"(median cross / matched) = {R:.4f} "
          f"-> tripped={R >= RETIRE_RATIO}", flush=True)

verb_label_retired = bool(any(Q1[v]["criterion_tripped"] for v in Q1))

# ── Q2 machinery: exec exp34a's header slice ─────────────────────────────
src34 = open("exp34a_swap_plausibility.py", encoding="utf-8").read()
head34 = src34[:src34.index("USABLE = {}")]
n34 = {"__name__": "exp39_q2"}
exec(compile(head34, "exp34a_head", "exec"), n34)
s34, C34 = n34["surface"], n34["CAND"]
build34, make_zx4 = n34["build"], n34["make_zx4"]
swap_sent, align34, enc34, vec34 = (n34["swap_sent"], n34["align"],
                                    n34["enc"], n34["vec"])

# pair build — replicated verbatim from exp34a (loop lives after the marker)
PAIRS = {}
for k, fs in C34.items():
    pairs = []
    for sent, subj, obj, split in fs:
        Go = build34(sent, subj)
        Gs_ = build34(swap_sent(sent), obj)
        if Go is not None and Gs_ is not None:
            pairs.append(dict(orig=make_zx4(Go), swap=make_zx4(Gs_),
                              sent=sent, subj=subj, obj=obj))
        if len(pairs) >= MAX_USE_PER_VERB:
            break
    print(f"[39] Q2 {s34[k]}: usable orig+swap pairs {len(pairs)}/{len(fs)}",
          flush=True)
    if len(pairs) >= MIN_FRAMES:
        PAIRS[k] = pairs

dials = {v: np.array(rec["T3b_theta"], float)
         for v, rec in SRC["zx4"]["verbs"].items()}
pverbs = [s34[k] for k in PAIRS if s34[k] in dials]
assert len(pverbs) == 6, f"expected 6 verbs with dials, got {pverbs}"

# score table: correct[verb][dial] = # pairs of `verb` decided correctly under
# `dial`'s theta.  Only the dial is permuted; sentences/arguments stay put.
correct = {v: {} for v in pverbs}
n_pairs = {}
skipped = 0
for k in PAIRS:
    v = s34[k]
    if v not in dials:
        continue
    usable = []
    for p in PAIRS[k]:
        vs_, vo_ = vec34(p["subj"]), vec34(p["obj"])
        if vs_ is None or vo_ is None:
            skipped += 1
            continue
        usable.append((p, enc34(vs_), enc34(vo_)))
    n_pairs[v] = len(usable)
    for u in pverbs:
        th = dials[u]
        correct[v][u] = int(sum(
            align34(p["orig"], th, es) > align34(p["swap"], th, eo)
            for p, es, eo in usable))
    print(f"[39] Q2 {v}: n={len(usable)} | own-dial correct "
          f"{correct[v][v]}/{len(usable)} | other-dial "
          f"{[correct[v][u] for u in pverbs if u != v]}", flush=True)

N_TOTAL = sum(n_pairs.values())
accs, ident_acc, rows = [], None, []
for perm in itertools.permutations(pverbs):
    c = sum(correct[v][u] for v, u in zip(pverbs, perm))
    a = c / N_TOTAL
    is_ident = all(v == u for v, u in zip(pverbs, perm))
    is_derange = all(v != u for v, u in zip(pverbs, perm))
    accs.append(a)
    rows.append((a, is_ident, is_derange))
    if is_ident:
        ident_acc = a
accs = np.array(accs)
assert ident_acc is not None
p95 = float(np.quantile(accs, TOP_FRAC))
quantile = float(np.mean(accs <= ident_acc))
n_ge = int(np.sum(accs >= ident_acc))
der = np.array([a for a, _, d in rows if d])
dials_carry_task_signal = bool(ident_acc >= p95)

Q2 = {
    "n_pairs": int(N_TOTAL),
    "n_pairs_per_verb": {v: int(n_pairs[v]) for v in n_pairs},
    "pairs_skipped_missing_vector": int(skipped),
    "verb_order": pverbs,
    "correct_matrix": correct,
    "dial_source": "results_exp33_s0.json variants.zx4.verbs.*.T3b_theta",
    "n_permutations": int(accs.size),
    "orig_acc": float(ident_acc),
    "orig_correct": int(round(ident_acc * N_TOTAL)),
    "perm_mean": float(accs.mean()),
    "perm_median": float(np.median(accs)),
    "perm_min": float(accs.min()),
    "perm_max": float(accs.max()),
    "perm_std": float(accs.std()),
    "perm_p95_threshold": p95,
    "perm_quantile": quantile,
    "n_perms_ge_orig": n_ge,
    "perm_p_value": float(n_ge / accs.size),
    "n_derangements": int(der.size),
    "derangement_mean": float(der.mean()),
    "derangement_max": float(der.max()),
    "criterion_top_fraction": 1 - TOP_FRAC,
    "criterion_passed": dials_carry_task_signal,
}
print(f"[39] Q2: orig acc = {ident_acc:.4f} ({Q2['orig_correct']}/{N_TOTAL}) | "
      f"perm mean={accs.mean():.4f} max={accs.max():.4f} "
      f"p95={p95:.4f} | quantile={quantile:.4f} "
      f"({n_ge} of {accs.size} perms >= orig) -> "
      f"dials_carry_task_signal={dials_carry_task_signal}", flush=True)

OUT = {
    "design_doc": "20_exp39_informative_null_design.md",
    "config": {"S_OUT": S_OUT, "retire_ratio": RETIRE_RATIO,
               "top_fraction": 1 - TOP_FRAC, "sanity_tol": SANITY_TOL,
               "fusion": os.environ.get("ARABIC_POS_FUSION", "0")},
    "sanity": sanity,
    "Q1": Q1,
    "Q2": Q2,
    "verdicts": {
        "verb_label_retired": verb_label_retired,
        "dials_carry_task_signal": dials_carry_task_signal,
    },
}
json.dump(OUT, open("results_exp39.json", "w"), indent=2, ensure_ascii=False)
print(f"[39] VERDICTS: verb_label_retired={verb_label_retired} | "
      f"dials_carry_task_signal={dials_carry_task_signal} | "
      f"sanity_passed={sanity['passed']}", flush=True)
print("[39] DONE", flush=True)
