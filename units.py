# -*- coding: utf-8 -*-
"""Unit-sentence demonstration for the 4 new modifier rules.
For each sentence prints: live (ENRICH off) fired rule, then enriched
(ENRICH on) fired rule + diagram string. Verifies cod == s."""
import sys, json, logging
CAP=[]
class Catch(logging.Handler):
    def emit(self, r):
        m=r.getMessage()
        if m.startswith("→"): CAP.append(m.replace("→","").strip())
import arabic_dep_reader as adr
from lambeq.backend.grammar import Ty
adr.logger.setLevel(logging.DEBUG); adr.logger.addHandler(Catch())
logging.getLogger("camel_test2").setLevel(logging.CRITICAL)
from camel_test2 import analyze_arabic_sentence_with_morph

UNITS = {
 "A. Attributive adjective (amod)": [
   "قرأ الولد الكتاب الجديد","الولد الطويل يقرأ الكتاب","كتب الطالب الدرس الجديد",
   "أكل الرجل الطعام اللذيذ","شرب الولد الماء البارد","رأى المعلم الطالب المجتهد",
   "فتح الرجل الباب الكبير","الطالبة المجتهدة تكتب الدرس","حمل العامل الصندوق الثقيل",
   "كسر الولد الكوب الزجاجي",
 ],
 "B. Idafa construct (nmod)": [
   "كتاب الولد جديد","فتح الولد باب البيت","قرأ الطالب كتاب النحو","باب البيت كبير",
   "سيارة الرجل سريعة","كتب المعلم درس اللغة","بيت الرجل كبير","حمل الولد حقيبة المدرسة",
   "قلم الطالب جديد","فتح المدير باب المكتب",
 ],
 "C. Post-verbal adverb (obl/advmod)": [
   "يجري الولد سريعا","ركض الولد كثيرا","يعمل المهندس جيدا","جاء الرجل مبكرا",
   "عاد الطالب متأخرا","نام الطفل قليلا","يكتب الطالب دائما","تكلم الرجل طويلا",
   "سافر الرجل بعيدا","ضحك الولد كثيرا",
 ],
 "D. Prepositional adjunct (case/obl)": [
   "جلس الولد في البيت","ذهب الرجل إلى المدرسة","يلعب الولد في الحديقة","نام الطفل على السرير",
   "كتب الطالب بالقلم","جلس الرجل على الكرسي","ذهب الولد إلى السوق","سار الرجل في الطريق",
   "رجع الولد من المدرسة","وقف الطالب في الصف",
 ],
}

def fire(sent):
    CAP.clear()
    t,a,s,r = analyze_arabic_sentence_with_morph(sent)
    d = adr.sentence_to_diagram_from_parse(t,a,s,r)
    return (CAP[-1] if CAP else "?"), d

S = Ty('s')
for cat, sents in UNITS.items():
    print("\n"+"="*74); print(cat); print("="*74)
    for sent in sents:
        adr.ENRICH_MODIFIERS = False
        live_rule, live_d = fire(sent)
        adr.ENRICH_MODIFIERS = True
        enr_rule, enr_d = fire(sent)
        adr.ENRICH_MODIFIERS = False
        ok = "cod=s OK" if enr_d.cod == S else f"BAD cod={enr_d.cod}"
        print(f"\nSENT: {sent}")
        print(f"  live(ENRICH off) rule: {live_rule}")
        print(f"  enriched rule        : {enr_rule}   [{ok}]")
        print(f"  enriched DIAGRAM     : {enr_d}")
sys.stdout.flush()
