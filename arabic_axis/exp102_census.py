# -*- coding: utf-8 -*-
"""exp102_census.py -- doc 99 (REGISTERED 8a02346): does any Arabic context SELECT by root alone or pattern alone?

CAMeL (built-in MSA db) analysis of PADT NOUNs, disambiguated by the PADT lemma; frozen Forms II-X masdar list;
P-SEL (light verb + masdar, >= 20 roots), R-SEL candidates (same root as governing verb, not masdar), joint
(absolute object), cohesion (descriptive).  Denominator: verbal selection links (NOUN child of VERB, obj, or obl with
a case child ب).  Writes census_exp102.json + gmasdar_sample_exp102.json (Rule 3).  Run with ~/camel_mac_env.
"""
import collections, json, os, random, re
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import exp94_common as C94
import exp100_census as CEN

DIAC = re.compile("[ً-ْٰـ]")
MASDAR = {"تَ1ْ2ِي3", "تَ1ْ2ِ3َة", "مُ1ا2َ3َة", "إِ1ْ2ا3", "إِ1ا3َة", "تَ1َ2ُّ3", "تَ1ا2ُ3", "اِنْ1ِ2ا3", "اِ1ْتِ2ا3",
          "اِسْتِ1ْ2ا3", "اِسْتِ1ا3َة"}
LIGHT = {"قام", "اجرى", "شن", "ابدى", "بذل", "واصل", "تابع", "باشر"}
CASE_END = "ًٌٍَُِْ"


def norm(s):
    s = DIAC.sub("", s or "")
    return re.sub("[أإآٱ]", "ا", s).split("_")[0]


def pat(p):
    p = (p or "").replace("ٱ", "ا")
    while p and p[-1] in CASE_END:
        p = p[:-1]
    return p


def main():
    an = Analyzer(MorphologyDB.builtin_db())
    cache = {}
    def analyse(form, lemma):
        k = (form, lemma)
        if k not in cache:
            keep = [a for a in an.analyze(form) if a.get("pos") == "noun" and norm(a.get("lex")) == norm(lemma)]
            roots = {a.get("root") for a in keep}
            if not keep:
                cache[k] = ("UNANALYSED", None, None)
            elif len(roots) > 1:
                cache[k] = ("AMBIGUOUS", None, None)
            else:
                cache[k] = ("OK", keep[0].get("root"), pat(keep[0].get("pattern")))
        return cache[k]
    vroot_cache = {}
    def verb_root(form, lemma):
        k = (form, lemma)
        if k not in vroot_cache:
            rs = {a.get("root") for a in an.analyze(form) if a.get("pos") == "verb" and norm(a.get("lex")) == norm(lemma)}
            vroot_cache[k] = rs.pop() if len(rs) == 1 else None
        return vroot_cache[k]
    stat = collections.Counter(); sel_links = 0
    psel, psel_roots, rsel, joint, cohesion = 0, set(), 0, 0, 0
    masdar_items = []
    for fn in CEN.FILES:
        path = os.path.join(CEN.DATA, fn)
        raw = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")]
        sent = []
        for c in raw + [[""]]:
            if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
                sent.append(c); continue
            if c == [""] or (len(c) == 1 and not c[0]):
                if sent:
                    kids = collections.defaultdict(list)
                    for t in sent:
                        kids[int(t[6])].append(t)
                    noun_info = {}
                    for t in sent:
                        if t[3] == "NOUN":
                            s_, r_, p_ = analyse(t[1], t[2]); stat[s_] += 1
                            noun_info[t[0]] = (s_, r_, p_)
                            if s_ == "OK" and p_ in MASDAR:
                                masdar_items.append({"form": t[1], "lemma": t[2], "root": r_, "pattern": p_})
                    for t in sent:
                        if t[3] != "NOUN" or noun_info[t[0]][0] != "OK":
                            continue
                        h = int(t[6])
                        if h == 0 or sent[h - 1][3] != "VERB":
                            continue
                        is_b = any(k[7] == "case" and norm(k[2]) == "ب" for k in kids[int(t[0])])
                        if not (t[7] == "obj" or (t[7].startswith("obl") and is_b)):
                            continue
                        sel_links += 1
                        _, r_, p_ = noun_info[t[0]]; v = sent[h - 1]; vl = norm(v[2])
                        is_m = p_ in MASDAR
                        if vl in LIGHT and is_m and (vl != "قام" or is_b):
                            psel += 1; psel_roots.add(r_)
                        vr = verb_root(v[1], v[2])
                        if vr and vr == r_:
                            if is_m:
                                joint += 1
                            else:
                                rsel += 1
                    ok = [(t[0], noun_info[t[0]]) for t in sent if t[3] == "NOUN" and noun_info[t[0]][0] == "OK"]
                    for (i, (_, r1, p1)), (j, (_, r2, p2)) in [(a, b) for k, a in enumerate(ok) for b in ok[k + 1:]]:
                        if r1 == r2 and p1 != p2:
                            cohesion += 1
                sent = []
    if len(psel_roots) < 20:
        psel_eff = 0
    else:
        psel_eff = psel
    out = {"doc": "99 (8a02346)", "noun_analysis": dict(stat),
           "unanalysed_rate": stat["UNANALYSED"] / sum(stat.values()), "ambiguous_rate": stat["AMBIGUOUS"] / sum(stat.values()),
           "verbal_selection_links": sel_links, "P_SEL": psel, "P_SEL_distinct_roots": len(psel_roots), "P_SEL_effective": psel_eff,
           "R_SEL_candidates": rsel, "joint_absolute_object": joint, "cohesion_pairs": cohesion,
           "share_P_SEL": psel_eff / sel_links, "share_R_SEL": rsel / sel_links,
           "nonnested_root_pattern": psel_eff / sel_links >= 0.05 and rsel / sel_links >= 0.05,
           "n_masdar_tokens": len(masdar_items), "provenance": C94.provenance()}
    rng = random.Random(102)
    uniq = list({(m["form"], m["lemma"]): m for m in masdar_items}.values())
    C94.write_new_json("gmasdar_sample_exp102.json", {"doc": "99 G-MASDAR sample", "items": rng.sample(uniq, 30)})
    C94.write_new_json("census_exp102.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
