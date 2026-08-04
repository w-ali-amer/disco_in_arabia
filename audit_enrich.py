# -*- coding: utf-8 -*-
"""Enrichment-mode audit: ENRICH_MODIFIERS=True. Measures how far the 4 rules
reach across the corpus (opt-in mode, NOT the live pipeline)."""
import sys, json, logging, traceback
OUT="audit_enrich.json"
CAP=[]
class Catch(logging.Handler):
    def emit(self,r):
        m=r.getMessage()
        if m.startswith("→"): CAP.append(m.replace("→","").strip())
import arabic_dep_reader as adr
adr.ENRICH_MODIFIERS=True    # <-- enrichment ON
adr.logger.setLevel(logging.DEBUG); adr.logger.addHandler(Catch())
logging.getLogger("camel_test2").setLevel(logging.CRITICAL)
from camel_test2 import analyze_arabic_sentence_with_morph
from lambeq.backend.grammar import Ty
S=Ty('s')
data=json.load(open("sentences.json"))
result={}
for split,items in data.items():
    per=[]
    for idx,item in enumerate(items):
        sent=item.get("sentence","")
        rule="ERR"; diagram=""; cods=False
        try:
            CAP.clear()
            t,a,st,r=analyze_arabic_sentence_with_morph(sent)
            d=adr.sentence_to_diagram_from_parse(t,a,st,r)
            rule=CAP[-1] if CAP else "NONE"
            diagram=str(d); cods=(d.cod==S)
        except Exception as e:
            rule="EXC"; diagram="EXC:"+repr(e)
        per.append({"idx":idx,"sentence":sent,"rule":rule,"diagram":diagram,"cods":cods})
    result[split]={"per_sentence":per}
    print(f"[{split}] done", flush=True)
json.dump(result,open(OUT,"w"),ensure_ascii=False)
print("WROTE",OUT,flush=True)
