# -*- coding: utf-8 -*-
"""exp120_data.py -- doc 120 (REGISTERED 984ae87): build the SRE-factorial sentence set and validate every form.

VSO "verb subject object"; subjects الرجل/الولد (m), المرأة/البنت (f), past verb agreeing (-ت for f); each object noun
in definite singular, dual (-ين / -تين) and plural.  Label = object sense class (0 concrete, 1 abstract).
G-FORMS: CAMeL must give the intended root (the singular's noun root) and number (s/d/p) for every form; failures
are dropped and listed.  Gold (root, pattern) = the first matching analysis; MLE (root, pattern) = CAMeL MLE top
analysis of the token in its sentence.  Patterns are normalised by stripping trailing case harakat.
Writes data_exp120.json (Rule 3).
"""
import json, os, re
from collections import Counter
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator

ROOT = os.path.dirname(os.path.abspath(__file__))
A = Analyzer(MorphologyDB.builtin_db())
MLE = MLEDisambiguator.pretrained()

NOUNS = {   # singular: (dual, plural)
    "الحجر": ("الحجرين", "الأحجار"), "الصندوق": ("الصندوقين", "الصناديق"), "الكرسي": ("الكرسيين", "الكراسي"),
    "الكيس": ("الكيسين", "الأكياس"), "الحقيبة": ("الحقيبتين", "الحقائب"), "الطاولة": ("الطاولتين", "الطاولات"),
    "السلة": ("السلتين", "السلال"), "الطفل": ("الطفلين", "الأطفال"), "الحبل": ("الحبلين", "الحبال"),
    "الشجرة": ("الشجرتين", "الأشجار"), "الورقة": ("الورقتين", "الأوراق"), "الخيط": ("الخيطين", "الخيوط"),
    "الغصن": ("الغصنين", "الأغصان"), "السلك": ("السلكين", "الأسلاك"), "التفاحة": ("التفاحتين", "التفاحات"),
    "الباب": ("البابين", "الأبواب"), "النافذة": ("النافذتين", "النوافذ"), "العلبة": ("العلبتين", "العلب"),
    "الدرج": ("الدرجين", "الأدراج"),
    "القضية": ("القضيتين", "القضايا"), "الدعوى": ("الدعويين", "الدعاوى"), "الشكوى": ("الشكويين", "الشكاوى"),
    "الضريبة": ("الضريبتين", "الضرائب"), "السعر": ("السعرين", "الأسعار"), "التقرير": ("التقريرين", "التقارير"),
    "الطلب": ("الطلبين", "الطلبات"), "الفكرة": ("الفكرتين", "الأفكار"), "المسؤولية": ("المسؤوليتين", "المسؤوليات"),
    "الأمانة": ("الأمانتين", "الأمانات"), "الهم": ("الهمين", "الهموم"), "الأمل": ("الأملين", "الآمال"),
    "المعنى": ("المعنيين", "المعاني"), "الخبر": ("الخبرين", "الأخبار"), "العلاقة": ("العلاقتين", "العلاقات"),
    "الصلة": ("الصلتين", "الصلات"), "الاتصال": ("الاتصالين", "الاتصالات"), "الحديث": ("الحديثين", "الأحاديث"),
    "الدراسة": ("الدراستين", "الدراسات"), "الرحلة": ("الرحلتين", "الرحلات"), "الموضوع": ("الموضوعين", "المواضيع"),
    "الحوار": ("الحوارين", "الحوارات"), "التحقيق": ("التحقيقين", "التحقيقات"), "النقاش": ("النقاشين", "النقاشات"),
    "الحساب": ("الحسابين", "الحسابات"), "المجال": ("المجالين", "المجالات"),
}
VERBS = {   # verb: (concrete nouns, abstract nouns)  -- doc 120 s6 (السلك replaces اللحم: no natural dual)
    "رفع": ("الحجر الصندوق الكرسي الكيس الحقيبة الطاولة السلة", "القضية الدعوى الشكوى الضريبة السعر التقرير الطلب"),
    "حمل": ("الصندوق الحقيبة الكيس الطفل السلة الكرسي الحجر", "الفكرة المسؤولية الأمانة الهم الأمل المعنى الخبر"),
    "قطع": ("الحبل الشجرة الورقة الخيط الغصن السلك التفاحة", "العلاقة الصلة الاتصال الحديث الدراسة الرحلة"),
    "فتح": ("الباب النافذة الصندوق العلبة الحقيبة الدرج", "الموضوع الحوار التحقيق النقاش الحساب المجال"),
}
SUBJ = {"الرجل": "m", "الولد": "m", "المرأة": "f", "البنت": "f"}
CASE = re.compile(r"[ً-ْ]+$")


def norm_pat(p):
    return CASE.sub("", p or "")


def noun_root(sg):
    """The root shared by the singular's and plural's noun analyses (most common among them); else the singular's."""
    roots = Counter(a["root"] for a in A.analyze(sg) if a.get("pos") == "noun" and a.get("num") == "s")
    plr = {a["root"] for a in A.analyze(NOUNS[sg][1]) if a.get("pos") == "noun" and a.get("num") == "p"}
    shared = Counter({r: c for r, c in roots.items() if r in plr})
    pick = shared or roots
    return pick.most_common(1)[0][0] if pick else None


def gold(form, root, num):
    for a in A.analyze(form):
        if a.get("root") == root and a.get("num") == num and a.get("pos") == "noun":
            return norm_pat(a["pattern"])
    return None


def main():
    out_p = os.path.join(ROOT, os.environ.get("EXP120_DATA", "data_exp120.json"))
    assert not os.path.exists(out_p), "Rule 3"
    forms, failed = {}, []
    for sg, (du, pl) in NOUNS.items():
        r = noun_root(sg)
        for form, num in ((sg, "s"), (du, "d"), (pl, "p")):
            pat = gold(form, r, num) if r else None
            if pat is None:
                failed.append({"lemma": sg, "form": form, "num": num, "root": r})
            else:
                forms[form] = {"lemma": sg, "num": num, "root": r, "pattern": pat}
    vgold = {}
    for v in VERBS:
        for g, vf in (("m", v), ("f", v + "ت")):
            an = [a for a in A.analyze(vf) if a.get("pos") == "verb" and a.get("asp") == "p" and a.get("vox") == "a"
                  and "\u0651" not in a.get("pattern", "")]                    # Form I active (no shadda), v2 fix
            vgold[vf] = {"root": an[0]["root"], "pattern": norm_pat(an[0]["pattern"])} if an else None
    items, mle_agree = [], Counter()
    for v, (conc, abst) in VERBS.items():
        for label, nl in ((0, conc.split()), (1, abst.split())):
            for sg in nl:
                for form in (sg,) + NOUNS[sg]:
                    if form not in forms:
                        continue
                    for subj, g in SUBJ.items():
                        vf = v if g == "m" else v + "ت"
                        toks = [vf, subj, form]
                        dis = MLE.disambiguate(toks)
                        mle = [{"root": d.analyses[0].analysis.get("root"), "pattern": norm_pat(d.analyses[0].analysis.get("pattern")),
                                "pos": d.analyses[0].analysis.get("pos")} for d in dis]
                        f = forms[form]
                        mle_agree["obj_root"] += mle[2]["root"] == f["root"]; mle_agree["obj_pattern"] += mle[2]["pattern"] == f["pattern"]
                        mle_agree["verb_root"] += bool(vgold[vf]) and mle[0]["root"] == vgold[vf]["root"]
                        mle_agree["verb_pattern"] += bool(vgold[vf]) and mle[0]["pattern"] == vgold[vf]["pattern"]
                        mle_agree["n"] += 1
                        items.append({"sentence": " ".join(toks), "verb": v, "subject": subj, "lemma": sg, "form": form,
                                      "num": f["num"], "label": label, "gold": {"verb": vgold[vf], "obj": {"root": f["root"], "pattern": f["pattern"]}},
                                      "mle": mle})
    n = mle_agree.pop("n")
    rates = {k: c / n for k, c in mle_agree.items()}
    per_verb = {v: {"concrete": len(c.split()), "abstract": len(a.split())} for v, (c, a) in VERBS.items()}
    pats = Counter((f["pattern"], f["num"]) for f in forms.values())
    json.dump({"doc": "120 (984ae87)", "items": items, "forms": forms, "verb_gold": vgold, "failed_forms": failed,
               "mle_agreement": rates, "roots_per_sense_per_verb": per_verb, "pattern_counts": {"%s|%s" % k: c for k, c in pats.items()}},
              open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("items", len(items), "forms", len(forms), "failed", len(failed), failed)
    print("MLE agreement", {k: round(v, 3) for k, v in rates.items()})
    print("verb gold", vgold)
    print("labels", Counter(i["label"] for i in items), "nums", Counter(i["num"] for i in items))


if __name__ == "__main__":
    main()
