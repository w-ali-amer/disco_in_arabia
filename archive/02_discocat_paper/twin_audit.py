"""Symbol audit on a TRUE multiset twin (exp15's audit used a scrambled pair)."""
import exp13_arabert_comparison as e
svo, vso = "الولد يقرا الكتاب", "يقرا الولد الكتاب"
diags = e.sentences_to_diagrams([svo, vso], log_interval=999)
for ns in (1, 2):
    ansatz = e.make_ansatz(1, ns)
    c = [ansatz(e._remove_cups(d)) for d in diags]
    s0 = {str(x) for x in c[0].free_symbols}
    s1 = {str(x) for x in c[1].free_symbols}
    print("n_s=%d shared=%d only_svo=%d only_vso=%d" % (ns, len(s0 & s1), len(s0 - s1), len(s1 - s0)))
    print("  shared:", sorted(s0 & s1))
