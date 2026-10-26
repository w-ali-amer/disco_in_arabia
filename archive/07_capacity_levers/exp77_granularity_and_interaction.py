#!/usr/bin/env python3
"""exp77 pre-freeze deliverables 2 and 3.

D2 GRANULARITY. How many distinct TRUE patterns hide inside each wazn_tag
family, and what the root x wazn grid looks like at tag-family granularity
versus true-pattern granularity. This is the table the freeze decision (doc 66
SS3: "R roots x W wazn ... R and W frozen at design freeze") is read from.

D3 TASK INTERACTION. Does the committed nonoccupational category label carry
root x wazn interaction, or is it approximately f(wazn)? Fitted on the ONLY
per-noun labels that exist (instrument_sb_bounds.CATEGORIES over the 24 frozen
HUMANS, doc 54 SS5.2, user-confirmed 2026-08-23). No labels are invented.
"""
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


INS = _load("instrument_sb_bounds", os.path.join(HERE, "instrument_sb_bounds.py"))
REC = json.load(open(os.path.join(HERE, "d4_roots_of_record.json"),
                     encoding="utf-8"))
ROWS = REC["nouns"]
H24 = REC["humans_24"]


def grid_stats(rows, root_key, wazn_key):
    """Incidence statistics for one (root granularity x wazn granularity)."""
    use = [r for r in rows if root_key(r) and wazn_key(r)]
    cells = defaultdict(list)
    for r in use:
        cells[(root_key(r), wazn_key(r))].append(r["noun"])
    roots = sorted({root_key(r) for r in use})
    wazns = sorted({wazn_key(r) for r in use})
    spans = Counter()
    for rt in roots:
        spans[len({w for (a, w) in cells if a == rt})] += 1
    multi = sum(v for k, v in spans.items() if k >= 2)
    return {
        "n_nouns_used": len(use),
        "R_roots": len(roots),
        "W_wazn": len(wazns),
        "cells_occupied": len(cells),
        "grid_size_RxW": len(roots) * len(wazns),
        "density_occupied_over_RxW": round(len(cells) / float(len(roots) * len(wazns)), 5),
        "mean_nouns_per_occupied_cell": round(len(use) / float(len(cells)), 3),
        "roots_spanning_ge2_wazn": multi,
        "roots_spanning_ge3_wazn": sum(v for k, v in spans.items() if k >= 3),
        "root_span_histogram": {str(k): v for k, v in sorted(spans.items())},
        "_cells": cells, "_wazns": wazns,
    }


def pair_table(cells, wazns, top=15):
    """Largest CLEAN wazn pairs: roots realised in exactly those two columns."""
    by_root = defaultdict(set)
    for (rt, w) in cells:
        by_root[rt].add(w)
    pairs = Counter()
    clean_pairs = Counter()
    for rt, ws in by_root.items():
        ws = sorted(ws)
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                pairs[(ws[i], ws[j])] += 1
        if len(ws) == 2:
            clean_pairs[(ws[0], ws[1])] += 1
    return (
        [{"pair": list(k), "n_roots_covering_pair": v}
         for k, v in pairs.most_common(top)],
        [{"pair": list(k), "n_roots_realised_in_exactly_this_pair": v}
         for k, v in clean_pairs.most_common(top)],
    )


def main():
    out = {}

    # ================================================== D2: GRANULARITY =====
    tag_of = lambda r: r["wazn_tag"]
    pat_of = lambda r: r["pattern"]
    root_of = lambda r: r["root"]
    root_strict = lambda r: r["root"] if r["root_complete_triliteral"] else None

    # --- how many true patterns hide inside each tag family ---------------
    fam = defaultdict(Counter)
    for r in ROWS:
        fam[r["wazn_tag"]][r["pattern"]] += 1
    family_table = []
    for tag in ["mu_participle", "nisba_yaa", "other", "faail",
                "faiil", "faaal_or_fa33aal"]:
        c = fam[tag]
        tot = sum(c.values())
        top = c.most_common(1)[0]
        family_table.append({
            "wazn_tag": tag,
            "n_nouns": tot,
            "n_distinct_true_patterns": len(c),
            "n_templatic_patterns": len([p for p in c if "NTWS" not in p]),
            "n_nouns_non_templatic_alNTWS": c.get("الNTWS", 0),
            "modal_pattern": top[0],
            "modal_pattern_n": top[1],
            "modal_share": round(top[1] / float(tot), 4),
            "patterns_with_1_noun": sum(1 for p, k in c.items() if k == 1),
            "top5": [{"pattern": p, "n": k} for p, k in c.most_common(5)],
        })
    out["d2_patterns_inside_each_tag_family"] = family_table
    out["d2_totals"] = {
        "n_nouns": len(ROWS),
        "W_at_tag_granularity": len(fam),
        "W_at_true_pattern_granularity": len({r["pattern"] for r in ROWS}),
        "W_true_pattern_templatic_only": len({r["pattern"] for r in ROWS
                                              if "NTWS" not in r["pattern"]}),
    }

    # --- tag <-> pattern cross-contamination, all six tags ----------------
    # doc 63 SS3.3 measured this for faail only (63 surface / 81 CAMeL / 62
    # intersection). Extended here to every tag, using the modal pattern of the
    # family as its morphological definition where no enumerated set exists.
    contam = []
    pattern_owner = {}
    for tag, c in fam.items():
        for p in c:
            pattern_owner.setdefault(p, Counter())
            pattern_owner[p][tag] += c[p]
    split_patterns = [{"pattern": p, "spread_over_tags": dict(t)}
                      for p, t in sorted(pattern_owner.items(),
                                         key=lambda kv: -sum(kv[1].values()))
                      if len(t) > 1]
    out["d2_true_patterns_split_across_tag_families"] = split_patterns[:20]
    out["d2_n_true_patterns_split_across_tags"] = len(split_patterns)

    # the faail row, reproducing doc 63 SS3.3 exactly, as a validity check
    surf = {r["noun"] for r in ROWS if r["wazn_tag"] == "faail"}
    cam = {r["noun"] for r in ROWS if r["faail_by_camel_pattern"]}
    out["d2_doc63_s3_3_reproduction"] = {
        "faail_surface_regex": len(surf), "faail_camel_pattern": len(cam),
        "intersection": len(surf & cam),
        "camel_only": sorted(cam - surf), "surface_only": sorted(surf - cam),
        "matches_doc63": (len(surf), len(cam), len(surf & cam)) == (63, 81, 62),
    }
    out["d2_nisba_tag_vs_pattern"] = {
        "nisba_by_surface_tag": sum(1 for r in ROWS if r["nisba_by_surface_tag"]),
        "nisba_by_true_pattern": sum(1 for r in ROWS if r["nisba_by_true_pattern"]),
        "intersection": sum(1 for r in ROWS if r["nisba_by_surface_tag"]
                            and r["nisba_by_true_pattern"]),
        "tag_only": sum(1 for r in ROWS if r["nisba_by_surface_tag"]
                        and not r["nisba_by_true_pattern"]),
        "pattern_only": sum(1 for r in ROWS if r["nisba_by_true_pattern"]
                            and not r["nisba_by_surface_tag"]),
    }

    # --- the grid, at both granularities, under two root definitions -------
    grids = {}
    for rname, rk in [("root_any_non_null", root_of),
                      ("root_complete_triliteral_only", root_strict)]:
        for wname, wk in [("wazn_tag_family", tag_of),
                          ("true_pattern", pat_of)]:
            g = grid_stats(ROWS, rk, wk)
            cells, wazns = g.pop("_cells"), g.pop("_wazns")
            allp, cleanp = pair_table(cells, wazns)
            g["largest_wazn_pairs_any"] = allp
            g["largest_clean_wazn_pairs_exactly_two"] = cleanp
            g["n_cells_with_ge2_nouns"] = sum(1 for v in cells.values() if len(v) >= 2)
            grids["%s | %s" % (rname, wname)] = g
    out["d2_grid"] = grids

    # --- transfer-relevant: cells usable for a factorial holdout ------------
    # doc 66 SS3 needs every root and every wazn to appear >= m_min times in
    # train, with held-out COMBINATIONS. The binding quantity is how many roots
    # span >= 2 wazn at the chosen granularity.
    out["d2_freeze_summary"] = [
        {"granularity": k, "R": v["R_roots"], "W": v["W_wazn"],
         "nouns": v["n_nouns_used"], "cells_occupied": v["cells_occupied"],
         "grid_RxW": v["grid_size_RxW"],
         "density": v["density_occupied_over_RxW"],
         "roots_spanning_ge2": v["roots_spanning_ge2_wazn"],
         "roots_spanning_ge3": v["roots_spanning_ge3_wazn"]}
        for k, v in grids.items()]

    # --- transfer capacity: the sub-grid a factorial holdout can actually use -
    def capacity(rk, wk):
        use = [r for r in ROWS if rk(r) and wk(r)]
        cells = defaultdict(list)
        for r in use:
            cells[(rk(r), wk(r))].append(r["noun"])
        by_root = defaultdict(set)
        for (rt, w) in cells:
            by_root[rt].add(w)
        multi = {rt: ws for rt, ws in by_root.items() if len(ws) >= 2}
        sub_cells = [k for k in cells if k[0] in multi]
        cols = Counter(w for ws in multi.values() for w in ws)
        return {
            "roots_spanning_ge2": len(multi),
            "nouns_in_sub_grid": sum(len(cells[k]) for k in sub_cells),
            "cells_in_sub_grid": len(sub_cells),
            "wazn_columns_in_sub_grid": len(cols),
            "roots_per_wazn_column": dict(cols.most_common(12)),
            "max_holdable_out_cells_keeping_every_root_seen": len(multi),
            "cells_with_ge2_nouns": sum(1 for k in sub_cells if len(cells[k]) >= 2),
        }

    out["d2_transfer_capacity"] = {
        "root_any | wazn_tag_family": capacity(root_of, tag_of),
        "root_complete_triliteral | wazn_tag_family": capacity(root_strict, tag_of),
        "root_any | true_pattern": capacity(root_of, pat_of),
        "root_complete_triliteral | true_pattern": capacity(root_strict, pat_of),
        "note": ("doc 66 SS3 trains on cells where EVERY root and EVERY wazn "
                 "appears >= m_min times and tests on held-out COMBINATIONS. "
                 "Only roots realised in >= 2 wazn can contribute a held-out "
                 "combination at all, so these are the binding numbers."),
    }

    # ============================================== D3: TASK INTERACTION ====
    # The ONLY per-noun category labels that exist.
    cats = INS.CATEGORIES
    humans = list(INS.HUMANS)
    hmap = {h["noun"]: h for h in H24}
    labelled = {
        "nonoccupational (the committed exp68/69/73/75 target)": {
            "n_labelled": len(humans), "n_positive": len(cats["nonoccupational"]["members"]),
            "members": sorted(cats["nonoccupational"]["members"]),
        },
        "tertiary": {"n_labelled": len(humans), "n_positive": 5,
                     "members": sorted(cats["tertiary"]["members"])},
        "uniformed": {"n_labelled": len(humans), "n_positive": 4,
                      "members": sorted(cats["uniformed"]["members"])},
        "manual": {"n_labelled": len(humans), "n_positive": 5,
                   "members": sorted(cats["manual"]["members"])},
    }
    out["d3_label_inventory"] = {
        "search": ("searched d4_model_check.json (veto/keep classes only, no "
                   "category field), d4_confirmed_inventory.json (noun / "
                   "wazn_tag / note only), candidate_human_nouns.json (no "
                   "category field), exp68_run.py / exp69_datagen.py / "
                   "exp73_run.py / exp75_run.py / exp76_run.py, and every *.py "
                   "and *.json in the repo for 'nonoccupational'"),
        "verdict": ("per-noun category labels exist ONLY over the 24 frozen "
                    "HUMANS. NO category label of any kind exists for the 519. "
                    "exp69_datagen.py:129-136 imports the members from "
                    "instrument_sb_bounds.CATEGORIES and asserts (4, 20); "
                    "exp73/exp75/exp76 all cite the same 24+category. The 44 "
                    "AraVec file carries vectors, not labels."),
        "labelled_subsets": labelled,
        "unassigned_by_doc54_s5_2": sorted(
            set(humans) - set().union(*[set(c["members"]) for c in cats.values()])),
    }

    # --- join the 24 to wazn / root ---------------------------------------
    y = {h: (1 if h in set(cats["nonoccupational"]["members"]) else 0)
         for h in humans}
    feat = {h: {"wazn_tag": hmap[h]["wazn_tag"], "root": hmap[h]["root"],
                "pattern": hmap[h]["pattern"]} for h in humans}
    base_major = max(Counter(y.values()).values()) / float(len(humans))

    def loo_majority(keyfn):
        """Leave-one-out majority-per-key, falling back to the global majority
        of the remaining 23 when the held-out noun's key is unseen."""
        correct, fell_back = 0, 0
        for h in humans:
            rest = [x for x in humans if x != h]
            k = keyfn(h)
            same = [x for x in rest if keyfn(x) == k and k is not None]
            if same:
                pred = Counter(y[x] for x in same).most_common(1)[0][0]
            else:
                pred = Counter(y[x] for x in rest).most_common(1)[0][0]
                fell_back += 1
            correct += int(pred == y[h])
        return correct / float(len(humans)), fell_back

    def resub_majority(keyfn):
        groups = defaultdict(list)
        for h in humans:
            groups[keyfn(h)].append(h)
        correct = 0
        for k, g in groups.items():
            m = Counter(y[x] for x in g).most_common(1)[0][0]
            correct += sum(1 for x in g if y[x] == m)
        return correct / float(len(humans)), len(groups)

    wf = lambda h: feat[h]["wazn_tag"]
    rf = lambda h: feat[h]["root"]
    pf = lambda h: feat[h]["pattern"]

    res = {}
    for nm, fn in [("wazn_tag", wf), ("true_pattern", pf), ("root", rf)]:
        acc_r, ngroups = resub_majority(fn)
        acc_l, fb = loo_majority(fn)
        res[nm] = {"resubstitution_acc": round(acc_r, 4),
                   "n_distinct_keys": ngroups,
                   "loo_acc": round(acc_l, 4),
                   "loo_fell_back_to_global_majority": fb}
    out["d3_classifiers"] = {
        "task": "predict nonoccupational membership over the 24 frozen HUMANS",
        "n": len(humans), "n_positive": sum(y.values()),
        "base_rate_majority_class": round(base_major, 4),
        "results": res,
        "per_noun": [{"noun": h, "y_nonoccupational": y[h],
                      "wazn_tag": feat[h]["wazn_tag"], "root": feat[h]["root"],
                      "pattern": feat[h]["pattern"]} for h in humans],
        "positives_by_wazn_tag": dict(Counter(feat[h]["wazn_tag"]
                                              for h in humans if y[h])),
        "all_by_wazn_tag": dict(Counter(feat[h]["wazn_tag"] for h in humans)),
        "root_uniqueness": {
            "n_nouns": len(humans),
            "n_distinct_roots": len({feat[h]["root"] for h in humans}),
            "n_roots_with_ge2_nouns": sum(
                1 for k, v in Counter(feat[h]["root"] for h in humans).items()
                if v >= 2 and k is not None),
        },
    }
    # the same two classifiers for the other three categories, for context
    other = {}
    for key in ["tertiary", "uniformed", "manual"]:
        yy = {h: (1 if h in set(cats[key]["members"]) else 0) for h in humans}
        saved = y.copy()
        y.clear(); y.update(yy)
        br = max(Counter(y.values()).values()) / float(len(humans))
        other[key] = {
            "base_rate": round(br, 4),
            "wazn_tag": {"resub": round(resub_majority(wf)[0], 4),
                         "loo": round(loo_majority(wf)[0], 4)},
            "root": {"resub": round(resub_majority(rf)[0], 4),
                     "loo": round(loo_majority(rf)[0], 4)},
        }
        y.clear(); y.update(saved)
    out["d3_other_categories_context"] = other

    op = os.path.join(HERE, "exp77_granularity_and_interaction.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", op)
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps(o["d2_totals"], ensure_ascii=False, indent=1))
    print(json.dumps(o["d2_patterns_inside_each_tag_family"],
                     ensure_ascii=False, indent=1))
    print(json.dumps(o["d2_freeze_summary"], ensure_ascii=False, indent=1))
    print(json.dumps(o["d2_doc63_s3_3_reproduction"]["matches_doc63"],
                     ensure_ascii=False))
    print(json.dumps(o["d3_classifiers"]["results"], ensure_ascii=False, indent=1))
    print("base rate", o["d3_classifiers"]["base_rate_majority_class"])
