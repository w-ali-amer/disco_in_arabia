from collections import Counter
import exp13_arabert_comparison as exp13
from lambeq import NumpyModel

SUBJ = "الرجل"
FRAMES_T = [("فتح", "الباب"), ("قرا", "الكتاب")]
sent_list = [f"{v} {SUBJ} {o}" for v, o in FRAMES_T]
diagrams = exp13.sentences_to_diagrams(sent_list, log_interval=999)
ansatz = exp13.make_ansatz(1, 1)
for (v, o), d in zip(FRAMES_T, diagrams):
    c = ansatz(exp13._remove_cups(d))
    kinds = Counter()
    for box in c.boxes:
        nm = str(getattr(box, "name", box))
        nd, nc = len(box.dom), len(box.cod)
        if nd == 0 and nc == 1: kinds["ket"] += 1
        elif nd == 1 and nc == 0: kinds["bra"] += 1
        elif nm == "H": kinds["H"] += 1
        elif nm.startswith("Rx("): kinds["Rx"] += 1
        elif nm.startswith("Rz("): kinds["Rz"] += 1
        elif nm.startswith("CRz("): kinds["CRz"] += 1
        elif nm.startswith("CRx("): kinds["CRx"] += 1
        else: kinds[nm[:12]] += 1
    print(f"[probe] {v}+{o}: {dict(kinds)}", flush=True)
    for box in c.boxes:
        nm = str(getattr(box, "name", box))
        if nm.startswith("CR") or (len(box.dom) == 2 and not nm.startswith("CR")):
            print(f"    2q: {nm[:60]}", flush=True)
