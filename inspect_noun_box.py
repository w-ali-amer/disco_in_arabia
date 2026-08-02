"""Print the gate structure of our circuits: which gates/symbols implement a
noun box? Determines whether noun word-states can be SOLVED for amplitude
encoding (any 1-qubit state = 3 Euler angles) in exp19."""
import exp13_arabert_comparison as e
diags = e.sentences_to_diagrams(["الولد يقرا الكتاب", "يقرا الولد الكتاب"], log_interval=999)
c = e.make_ansatz(1, 1)(e._remove_cups(diags[0]))
print("TYPE:", type(c))
try:
    for b in c.boxes:
        print("BOX:", repr(b))
except Exception as ex:
    print("boxes failed:", ex)
    print(repr(c))
