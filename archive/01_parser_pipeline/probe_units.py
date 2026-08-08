# -*- coding: utf-8 -*-
"""Probe: dump Stanza dep trees + fired rule for the 40 candidate unit sentences."""
import sys, json, logging

# capture the "→ XXX" rule logs from arabic_dep_reader
CAPTURED = []
class RuleCatcher(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("→"):  # arrow
            CAPTURED.append(msg)

import arabic_dep_reader as adr
adr.logger.setLevel(logging.DEBUG)
adr.logger.addHandler(RuleCatcher())

from camel_test2 import analyze_arabic_sentence_with_morph

UNITS = {
 "adjective": [
   "قرأ الولد الكتاب الجديد",
   "الولد الطويل يقرأ الكتاب",
   "كتب الطالب الدرس الجديد",
   "أكل الرجل الطعام اللذيذ",
   "شرب الولد الماء البارد",
   "رأى المعلم الطالب المجتهد",
   "فتح الرجل الباب الكبير",
   "الطالبة المجتهدة تكتب الدرس",
   "حمل العامل الصندوق الثقيل",
   "كسر الولد الكوب الزجاجي",
 ],
 "idafa": [
   "كتاب الولد جديد",
   "فتح الولد باب البيت",
   "قرأ الطالب كتاب النحو",
   "باب البيت كبير",
   "سيارة الرجل سريعة",
   "كتب المعلم درس اللغة",
   "بيت الرجل كبير",
   "حمل الولد حقيبة المدرسة",
   "قلم الطالب جديد",
   "فتح المدير باب المكتب",
 ],
 "adverb": [
   "يجري الولد سريعا",
   "ركض الولد كثيرا",
   "يعمل المهندس جيدا",
   "جاء الرجل مبكرا",
   "عاد الطالب متأخرا",
   "نام الطفل قليلا",
   "يكتب الطالب دائما",
   "تكلم الرجل طويلا",
   "سافر الرجل بعيدا",
   "ضحك الولد كثيرا",
 ],
 "pp": [
   "جلس الولد في البيت",
   "ذهب الرجل إلى المدرسة",
   "يلعب الولد في الحديقة",
   "نام الطفل على السرير",
   "كتب الطالب بالقلم",
   "جلس الرجل على الكرسي",
   "ذهب الولد إلى السوق",
   "سار الرجل في الطريق",
   "رجع الولد من المدرسة",
   "وقف الطالب في الصف",
 ],
}

for cat, sents in UNITS.items():
    print("="*70)
    print("CATEGORY:", cat)
    print("="*70)
    for s in sents:
        CAPTURED.clear()
        tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(s)
        diag = adr.sentence_to_diagram_from_parse(tokens, analyses, structure, roles)
        rule = CAPTURED[-1] if CAPTURED else "?NONE"
        print("\nSENT:", s)
        print("  structure:", structure)
        print("  roles: subj=%s verb=%s obj=%s pred=%s root=%s" % (
            roles.get('subject'), roles.get('verb'), roles.get('object'),
            roles.get('predicate_idx'), roles.get('root')))
        for i,a in enumerate(analyses):
            print("    [%d] %-12s upos=%-6s deprel=%-12s head=%s" % (
                i, a['text'], a['upos'], a['deprel'], a['head']))
        print("  depgraph:", dict(roles.get('dependency_graph', {})))
        print("  FIRED:", rule)
        print("  DIAGRAM:", str(diag))
sys.stdout.flush()
