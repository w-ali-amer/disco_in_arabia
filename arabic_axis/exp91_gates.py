# -*- coding: utf-8 -*-
"""exp91_gates.py -- doc 84 SS3: self-emitting, hash-pinned gate artifacts, each
with a designed-to-fail case shown FAILING. A gate PASSES iff its real
assertions all hold AND its designed-to-fail assertion FAILS.

    python3 exp91_gates.py {a91render,a91units,vsuite,all} [--outdir exp91_gates]
            [--units-dir exp91_ckpt] [--seeds 0 | --seeds all]

a91render  (primary, n12) and (confirm, n5): train AND val CONFLICT sets equal
           exp90's k_nat sets event for event (plus full item-list equality,
           stronger); the committed exp91 files equal a fresh rebuild.
           DTF: a rendering with render seed + 1 must NOT match.
a91units   every learner x seed unit at those two cells (from --units-dir,
           produced by exp91_run.py) reproduces exp90_ckpt/<draw>_<arm>_knat_s<seed>
           on conflict_acc, consistent_acc, pred_voc EXACTLY. --seeds 0 is the
           Stage-1 subset (scope recorded in the artifact); --seeds all is the
           full anchor set (6 x 12 x 2) re-verified from the grid's own units.
           DTF: the seed-0 exp91 unit vs exp90's seed-1 unit must mismatch.
vsuite     exp90's twin / balance / construction / encoder checks on every exp91
           rendering; per-dose no-double-rendering + held-out leak; exact counts;
           nestedness CONFLICT(n) subset CONFLICT(n') for n < n' (train and val).
           DTF: wrong-pair twins (T1) and a nesting violation.
"""
import argparse
import copy
import json
import os
import sys

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

import exp90_gates as g90                                  # noqa: E402
from exp90_common import canonical_sha256, provenance, write_new_json  # noqa: E402
import exp91_datagen as dg                                 # noqa: E402
from exp91_run import LEARNERS                             # noqa: E402

FILES91 = {d: dg.DRAWS[d]["out"] for d in dg.DRAWS}
FILES90 = {d: dg.DRAWS[d]["exp90"] for d in dg.DRAWS}
ANCHORS = [("primary", 12), ("confirm", 5)]
FIELDS = ("conflict_acc", "consistent_acc", "pred_voc")     # registered (doc 84 SS3)
SUPP = ("params_final", "best_epoch", "val_acc", "pred_unvoc")  # supplementary, reported


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def _prov():
    p = provenance(extra_files=list(FILES91.values()) + list(FILES90.values()))
    p["design_doc_91"] = dg.DESIGN_DOC_91
    return p


def conflict_sets(items):
    out = {"train": set(), "val": set()}
    for i in items:
        if i["item_type"] == "CONFLICT":
            out[i["split"]].add(i["base_id"])
    return out


# ============================================================ A91-render
def render_compare(items91, items90):
    c91, c90 = conflict_sets(items91), conflict_sets(items90)
    r = {s: {"n_exp91": len(c91[s]), "n_exp90": len(c90[s]),
             "equal": c91[s] == c90[s],
             "only_exp91": sorted(c91[s] - c90[s])[:5],
             "only_exp90": sorted(c90[s] - c91[s])[:5]} for s in ("train", "val")}
    r["full_item_lists_equal"] = items91 == items90
    r["passed"] = bool(r["train"]["equal"] and r["val"]["equal"])
    return r


def gate_a91render():
    real = {}
    for draw, n in ANCHORS:
        b91, b90 = _load(FILES91[draw]), _load(FILES90[draw])
        fresh = dg.build(draw)
        r = render_compare(b91["train_sets"]["n%d" % n], b90["train_sets"]["knat"])
        r["exp90_knat_counts"] = b90["meta"]["counts"]["knat"]
        r["exp91_counts"] = b91["meta"]["counts"]["n%d" % n]
        r["committed_file_equals_fresh_rebuild"] = (
            fresh["meta"]["data_sha256"] == b91["meta"]["data_sha256"])
        r["test_items_equal_exp90"] = b91["test"] == b90["test"]
        r["passed"] = bool(r["passed"] and r["committed_file_equals_fresh_rebuild"]
                           and r["test_items_equal_exp90"])
        real["%s_n%d" % (draw, n)] = r
    ok = all(v["passed"] for v in real.values())
    dtf = {}
    for draw, n in ANCHORS:
        rs = dg.RENDER_SEEDS[dg.DRAWS[draw]["base"]] + 1
        bad = dg.build(draw, render_seed=rs, check_registered=False)
        b90 = _load(FILES90[draw])
        d = render_compare(bad["train_sets"]["n%d" % n], b90["train_sets"]["knat"])
        d["render_seed_used"] = rs
        dtf["%s_n%d_render_seed_%d" % (draw, n, rs)] = d
    dtf_shown = all(not v["passed"] for v in dtf.values())
    return {"gate": "A91-render", "anchors": real, "real_all_pass": bool(ok),
            "designed_to_fail": dtf, "designed_to_fail_shown_failing": dtf_shown,
            "passed": bool(ok and dtf_shown), "provenance": _prov()}


# ============================================================= A91-units
def unit_compare(u91, u90):
    per = {f: u91[f] == u90[f] for f in FIELDS}
    sup = {f: u91[f] == u90[f] for f in SUPP}
    return {"fields_equal": per, "exact": all(per.values()),
            "supplementary_equal": sup, "supplementary_exact": all(sup.values()),
            "exp91": {"conflict_acc": u91["conflict_acc"],
                      "consistent_acc": u91["consistent_acc"]},
            "exp90": {"conflict_acc": u90["conflict_acc"],
                      "consistent_acc": u90["consistent_acc"]},
            "pred_voc_hamming": sum(a != b for a, b in zip(u91["pred_voc"], u90["pred_voc"])),
            "exp91_payload_sha256": u91["payload_sha256"],
            "exp90_payload_sha256": u90["payload_sha256"]}


def gate_a91units(units_dir, seeds):
    rows, missing = {}, []
    for draw, n in ANCHORS:
        for arm in LEARNERS:
            for s in seeds:
                p91 = os.path.join(units_dir, "%s_%s_n%d_s%d.json" % (draw, arm, n, s))
                p90 = os.path.join("exp90_ckpt", "%s_%s_knat_s%d.json" % (draw, arm, s))
                if not os.path.exists(p91):
                    missing.append(p91)
                    continue
                u91, u90 = _load(p91), _load(p90)
                assert u91["max_epochs"] == u90["max_epochs"] == 150, p91
                r = unit_compare(u91, u90)
                r.update({"exp91_unit": p91, "exp90_unit": p90})
                rows["%s_%s_n%d_s%d" % (draw, arm, n, s)] = r
    n_exact = sum(r["exact"] for r in rows.values())
    expected = len(ANCHORS) * len(LEARNERS) * len(seeds)
    ok = bool(not missing and len(rows) == expected and n_exact == expected)
    # designed-to-fail: exp91 seed-0 unit vs exp90's seed-1 unit
    dtf = {}
    for draw, n in ANCHORS:
        for arm in LEARNERS:
            p91 = os.path.join(units_dir, "%s_%s_n%d_s0.json" % (draw, arm, n))
            p90 = os.path.join("exp90_ckpt", "%s_%s_knat_s1.json" % (draw, arm))
            if os.path.exists(p91):
                r = unit_compare(_load(p91), _load(p90))
                r.update({"exp91_unit": p91, "exp90_unit_WRONG_SEED": p90})
                dtf["%s_%s_s0_vs_exp90_s1" % (draw, arm)] = r
    # exp90 convention: the DTF is "shown failing" iff the gate's own pass
    # criterion (every compared unit exact) FAILS on the wrong-seed data.
    # Individual wrong-seed pairs may coincide (CL-MLP has only 6 distinct
    # prediction strings over 12 exp90 seeds at confirm knat); they are counted.
    n_dtf_exact = sum(v["exact"] for v in dtf.values())
    dtf_shown = bool(dtf) and n_dtf_exact < len(dtf)
    dtf_summary = {"n_pairs": len(dtf), "n_pairs_coinciding_on_registered_fields": n_dtf_exact,
                   "coinciding_pairs": [k for k, v in dtf.items() if v["exact"]],
                   "n_pairs_coinciding_on_supplementary": sum(v["supplementary_exact"] for v in dtf.values()),
                   "criterion_applied_to_wrong_seed_data_passes": n_dtf_exact == len(dtf)}
    scope = ("stage1_subset_seed0" if list(seeds) == [0] else
             "full_anchor_set" if list(seeds) == list(range(12)) else
             "seeds_%s" % ",".join(map(str, seeds)))
    return {"gate": "A91-units", "scope": scope, "seeds": list(seeds),
            "fields_compared_exactly": list(FIELDS), "units_dir": units_dir,
            "n_expected": expected, "n_compared": len(rows), "n_exact": n_exact,
            "missing": missing, "rows": rows, "real_all_pass": ok,
            "n_supplementary_exact": sum(r["supplementary_exact"] for r in rows.values()),
            "designed_to_fail_summary": dtf_summary,
            "designed_to_fail": dtf, "designed_to_fail_shown_failing": dtf_shown,
            "passed": bool(ok and dtf_shown), "provenance": _prov()}


# ================================================================ V-suite
def renderings_ok(blob, draw):
    """Per-dose: each event rendered once, same event set across doses, no
    held-out leak, exact counts; nestedness across the ladder."""
    fails = []
    base_ids = None
    for cond, its in blob["train_sets"].items():
        seen = {}
        for i in its:
            seen.setdefault(i["base_id"], []).append(i["item_type"])
        dup = [b for b, v in seen.items() if len(v) != 1]
        if dup:
            fails.append((cond, "event rendered %d times" % len(dup), dup[:3]))
        if base_ids is None:
            base_ids = set(seen)
        elif set(seen) != base_ids:
            fails.append((cond, "event set differs across doses"))
        n = int(cond[1:])
        cs = conflict_sets(its)
        if len(cs["train"]) != n:
            fails.append((cond, "n_conflict_train %d != %d" % (len(cs["train"]), n)))
        if len(cs["val"]) != dg.EXPECTED_VAL[draw][n]:
            fails.append((cond, "n_conflict_val %d != registered %d"
                          % (len(cs["val"]), dg.EXPECTED_VAL[draw][n])))
    held = set(blob["meta"]["heldout_inventory"])
    leak = [i["id"] for its in blob["train_sets"].values() for i in its
            if set(i["referents"]) & held]
    if leak:
        fails.append(("heldout_leak", leak[:3]))
    nest = {}
    ladder = sorted(int(c[1:]) for c in blob["train_sets"])
    cs = {n: conflict_sets(blob["train_sets"]["n%d" % n]) for n in ladder}
    for a, b in zip(ladder, ladder[1:]):
        tr = cs[a]["train"] < cs[b]["train"]            # strict: counts differ
        va = cs[a]["val"] <= cs[b]["val"]
        nest["n%d<n%d" % (a, b)] = {"train_strict_subset": tr, "val_subset": va}
        if not (tr and va):
            fails.append(("nesting", a, b, tr, va))
    nested = all(v["train_strict_subset"] and v["val_subset"] for v in nest.values())
    return {"n_fail": len(fails), "fails": fails[:5], "nestedness": nest,
            "doses_nested": nested, "passed": not fails}


def gate_vsuite():
    per, ok = {}, True
    for draw, f in FILES91.items():
        blob = _load(f)
        r = {"T1_twin_circuit_identity_and_label_flip": g90.t1_twin_identity(blob),
             "role_balance": g90.role_balance(blob),
             "renderings_counts_and_nestedness": renderings_ok(blob, draw),
             "conf_is_conflict_rendering_of_other_twin": g90.conf_equals_other_twin(blob),
             "encoder_crosscheck_all_surfaces": g90.encoder_crosscheck_all(blob),
             "counts": blob["meta"]["counts"]}
        r["passed"] = all(v["passed"] for k, v in r.items() if k != "counts")
        ok &= r["passed"]
        per[f] = r
    blob = _load(FILES91["primary"])
    # (a) wrong-pair twins
    bad = copy.deepcopy(blob)
    confs = [i for i in bad["test"] if i["item_type"] == "CONFLICT"]
    tw = [i["twin_of"] for i in confs]
    for i, t in zip(confs, tw[1:] + tw[:1]):
        i["twin_of"] = t
    dtf_a = g90.t1_twin_identity(bad, seeds=(0,))
    # (b) nesting violation: n14 loses one of n12's CONFLICT train events
    #     (rendered CONSISTENT) and gains a fresh one -- counts stay exact
    #     (genuine renderings taken from the n5 / n72 lists, same index order)
    bad2 = copy.deepcopy(blob)
    T = bad2["train_sets"]
    c5 = conflict_sets(T["n5"])["train"]
    c12 = conflict_sets(T["n12"])["train"]
    c14 = conflict_sets(T["n14"])["train"]
    drop = next(k for k, i in enumerate(T["n14"])
                if i["base_id"] in c12 - c5)           # CONSISTENT at n5
    add = next(k for k, i in enumerate(T["n72"])
               if i["split"] == "train" and i["item_type"] == "CONFLICT"
               and i["base_id"] not in c14)
    T["n14"][drop] = copy.deepcopy(T["n5"][drop])
    T["n14"][add] = copy.deepcopy(T["n72"][add])
    assert T["n14"][drop]["item_type"] == "CONSISTENT"
    dtf_b = renderings_ok(bad2, "primary")
    dtf = {"wrong_pair_twins_T1": dtf_a, "nesting_violation_n12_n14": dtf_b}
    dtf_shown = all(not v["passed"] for v in dtf.values())
    return {"gate": "V-suite-91", "per_draw": per, "real_all_pass": bool(ok),
            "designed_to_fail": dtf, "designed_to_fail_shown_failing": dtf_shown,
            "passed": bool(ok and dtf_shown), "provenance": _prov()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gate", choices=["a91render", "a91units", "vsuite", "all"])
    ap.add_argument("--outdir", default="exp91_gates")
    ap.add_argument("--units-dir", default="exp91_ckpt")
    ap.add_argument("--seeds", default="0", help="'0' (Stage 1) or 'all'")
    ap.add_argument("--tag", default="",
                    help="artifact name suffix, e.g. _full for --seeds all")
    args = ap.parse_args()
    seeds = list(range(12)) if args.seeds == "all" else \
        [int(x) for x in args.seeds.split(",")]
    names = ["a91render", "vsuite", "a91units"] if args.gate == "all" else [args.gate]
    fns = {"a91render": gate_a91render, "vsuite": gate_vsuite,
           "a91units": lambda: gate_a91units(args.units_dir, seeds)}
    rc = 0
    for n in names:
        path = os.path.join(args.outdir, "%s%s.json" % (n, args.tag if n == "a91units" else ""))
        if os.path.exists(path):
            raise SystemExit("refusing to overwrite %s (use a fresh --outdir)" % path)
        r = fns[n]()
        sha = write_new_json(path, r)
        print("[gate %-9s] %s  real=%s  designed-to-fail shown failing=%s  -> %s (%s)"
              % (n, "PASS" if r["passed"] else "FAIL", r["real_all_pass"],
                 r["designed_to_fail_shown_failing"], path, sha[:12]), flush=True)
        rc |= (not r["passed"])
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
