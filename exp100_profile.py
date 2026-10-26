# -*- coding: utf-8 -*-
"""exp100_profile.py -- doc 97 §10 amendment A1 (92174d6): noun AGREEMENT PROFILE from linked readers.

x = (G_agr, N_agr, Definite, Case): G_agr = majority Gender over all linked readers; N_agr = majority Number over
linked readers EXCLUDING R-VSO (own UD Number if only R-VSO readers); ties -> noun excluded.  G-AGR gate (>= 0.95).
Writes profile_exp100.json (Rule 3). Reuses the frozen classifier of exp100_census.py.
"""
import collections, json, os
import exp94_common as C94
import exp100_census as CEN


def majority(vals):
    c = collections.Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def main():
    links = collections.defaultdict(list)           # noun key -> [(r, reader feats)]
    nouns = {}
    for fn in CEN.FILES:
        for si, sent in enumerate(CEN.sentences(os.path.join(CEN.DATA, fn))):
            kids = collections.defaultdict(list)
            for t in sent:
                kids[t["head"]].append(t)
            for t in sent:
                r, tgt = CEN.classify(t, kids[t["id"]], sent)
                if r and tgt is not None and r not in CEN.REMOTE:
                    key = (fn, si, tgt["id"])
                    nouns[key] = tgt
                    links[key].append((r, t["feats"]))
    joint, ties = collections.Counter(), 0
    agree = {"G": [0, 0], "N": [0, 0]}
    for key, L in links.items():
        n = nouns[key]
        gs = [f.get("Gender", "∅") for _, f in L]
        ns = [f.get("Number", "∅") for r, f in L if r != "R-VSO"]
        if len(L) >= 2:
            agree["G"][0] += 1; agree["G"][1] += len(set(gs)) == 1
        if len(ns) >= 2:
            agree["N"][0] += 1; agree["N"][1] += len(set(ns)) == 1
        g = majority(gs)
        nn = majority(ns) if ns else n["feats"].get("Number", "∅")
        if g is None or nn is None:
            ties += 1
            continue
        x = (g, nn, n["feats"].get("Definite", "∅"), n["feats"].get("Case", "∅"))
        for r, f in L:
            joint[(x, r, tuple(f.get(k, "∅") for k in CEN.READS[r]))] += 1
    g_agr = {k: (v[1] / v[0] if v[0] else None) for k, v in agree.items()}
    gate = all(v is not None and v >= 0.95 for v in g_agr.values())
    # doc 97 §6 empirical 2->1 prior: G_agr x {Sing, non-Sing}; primary N = N_agr, secondary N = own UD Number
    prior_agr, prior_own = collections.Counter(), collections.Counter()
    for key, L in links.items():
        gs = [f.get("Gender", "∅") for _, f in L]; ns = [f.get("Number", "∅") for r, f in L if r != "R-VSO"]
        g = majority(gs); own = nouns[key]["feats"].get("Number", "∅")
        nn = majority(ns) if ns else own
        if g in ("Masc", "Fem") and nn in ("Sing", "Dual", "Plur"):
            prior_agr[(g, "Sing" if nn == "Sing" else "nonSing")] += 1
        if g in ("Masc", "Fem") and own in ("Sing", "Dual", "Plur"):
            prior_own[(g, "Sing" if own == "Sing" else "nonSing")] += 1
    out = {"doc": "97 §10 A1 (92174d6)", "n_profiled_nouns": len(links), "ties_excluded": ties,
           "G_AGR": {"G": g_agr["G"], "N": g_agr["N"], "n_G": agree["G"][0], "n_N": agree["N"][0], "pass": gate},
           "linked_joint": [{"x": list(x), "r": r, "a": list(a), "n": c} for (x, r, a), c in sorted(joint.items())],
           "prior_2to1_agr": {"|".join(k): v for k, v in prior_agr.items()},
           "prior_2to1_own": {"|".join(k): v for k, v in prior_own.items()},
           "provenance": C94.provenance()}
    C94.write_new_json("profile_exp100.json", out)
    print(json.dumps({k: out[k] for k in ("n_profiled_nouns", "ties_excluded", "G_AGR", "prior_2to1_agr", "prior_2to1_own")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
