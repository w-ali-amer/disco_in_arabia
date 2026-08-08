"""Exp 23 Phase 0: offline CAMeL root (جذر) extraction for the full dataset
vocabulary + root-family survey. Uses the morphological 'root' field, NOT the
dependency head (known naming collision). Output: exp23_roots.json.
"""
import json, time
from collections import defaultdict

t0 = time.time()
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

data = json.load(open("sentences.json", encoding="utf-8"))
words = sorted({w for split in data.values() for d in split
                for w in d["sentence"].split()})
print(f"[23p0] {len(words)} unique tokens across all splits", flush=True)

analyzer = Analyzer(MorphologyDB.builtin_db())

out, missing = {}, []
for w in words:
    cands = [w] + ([w[2:]] if w.startswith("ال") else [])
    roots = defaultdict(int)
    for c in cands:
        try:
            for a in analyzer.analyze(c):
                r = a.get("root", "")
                if r and r not in ("NOAN", "NTWS"):
                    roots[r] += 1
        except Exception:
            pass
    if roots:
        top = max(roots, key=roots.get)
        out[w] = {"top": top, "candidates": dict(roots)}
    else:
        missing.append(w)

fams = defaultdict(list)
for w, info in out.items():
    fams[info["top"]].append(w)
multi = {r: ws for r, ws in fams.items() if len(ws) >= 2}
json.dump({"words": out, "missing": missing,
           "families_2plus": multi},
          open("exp23_roots.json", "w"), ensure_ascii=False, indent=1)
print(f"[23p0] roots found for {len(out)}/{len(words)} "
      f"(missing {len(missing)}: {missing[:10]})", flush=True)
print(f"[23p0] root families with >=2 members in existing vocab: "
      f"{len(multi)}", flush=True)
for r, ws in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:15]:
    print(f"[23p0]   {r}: {' '.join(ws)}", flush=True)
print(f"[23p0] DONE in {time.time()-t0:.1f}s", flush=True)
