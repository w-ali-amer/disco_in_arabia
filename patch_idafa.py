"""Apply the reviewer's tightening #1: mudaf must lack the definite article ال.
True idafa never carries ال on the head noun (كتاب الولد yes, الولد الكتاب never).
Kills the 4 false-idafa REJECTs (#1,#6,#9,#35) at zero true-idafa cost.
Caveat documented: adjectival (lafziyya) idafa is the known exception, absent
from all current datasets."""
import io

PATH = "arabic_dep_reader.py"
src = io.open(PATH, encoding="utf-8").read()
old = """            if dr in ('nmod', 'nmod:poss') and da['upos'] in ('NOUN', 'PROPN') \\
                    and di == head_idx + 1:
                idafa_idx = di
                break"""
new = """            if dr in ('nmod', 'nmod:poss') and da['upos'] in ('NOUN', 'PROPN') \\
                    and di == head_idx + 1 \\
                    and not tokens[head_idx].startswith('\\u0627\\u0644'):
                # true idafa: the mudaf never carries the definite article
                # (reviewer tightening #1; adjectival idafa is the documented
                # exception, absent from current datasets)
                idafa_idx = di
                break"""
n = src.count(old)
assert n == 1, f"pattern found {n} times"
io.open(PATH, "w", encoding="utf-8").write(src.replace(old, new))
print("IDAFA-FILTER PATCHED")
