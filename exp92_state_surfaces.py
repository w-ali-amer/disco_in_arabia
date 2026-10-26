# -*- coding: utf-8 -*-
"""exp92_state_surfaces.py -- doc 87 §2: state-token surfaces (participle of the INTRANSITIVE member).

For each frozen pair (wazn_b1/frozen_lexicon_v1.json), the intransitive member is VII for ANTI and I for CAUS.
Its Participle is looked up in Arabic VerbNet (~/ext/Arabic-Verbnet, commit recorded) by the member name
(diacritised perfect + final fatha, hamzat-wasl written as إ as AVN does). If AVN has none, source = "none" and the
state surface falls back to the bare lemma (recorded). Surfaces are cosmetic: the circuit sees a state ID.
"""
import json, glob, os, subprocess, unicodedata, hashlib
import xml.etree.ElementTree as ET
AVN = os.path.expanduser("~/ext/Arabic-Verbnet")
FZ = "wazn_b1/frozen_lexicon_v1.json"
OUT = "exp92_state_surfaces.json"
assert not os.path.exists(OUT), "Rule 3"
H = "ًٌٍَُِّْٰ"
def bare(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(c for c in s if c not in H)
    for a in "ٱإآأ":
        s = s.replace(a, "ا")
    return s
parts = {}
for fp in glob.glob(os.path.join(AVN, "*.xml")):
    for m in ET.parse(fp).getroot().iter("MEMBER"):
        p = [x.get("name") for x in m.findall("Participle")]
        if p:
            parts.setdefault(bare(m.get("name")), []).extend(p)
fz = json.load(open(FZ, encoding="utf-8"))
out = {}
for alt, key in (("I-VII", "VII"), ("I-IV", "I")):
    for x in fz["pairs"][alt]:
        intr = x["VII"] if key == "VII" else x["I"]
        cand = parts.get(bare(intr), [])
        out[x["pair"]] = {"alternation": alt, "intransitive_lex": intr,
                          "state_surface": cand[0] if cand else bare(intr),
                          "source": "AVN_participle" if cand else "none_bare_lemma",
                          "all_avn_participles": sorted(set(cand))}
meta = {"avn_commit": subprocess.check_output(["git", "-C", AVN, "rev-parse", "HEAD"], text=True).strip(),
        "frozen_sha256": hashlib.sha256(open(FZ, "rb").read()).hexdigest()}
json.dump({"meta": meta, "states": out}, open(OUT, "w"), ensure_ascii=False, indent=1, sort_keys=True)
for k, v in out.items():
    print(k, v["state_surface"], v["source"])
