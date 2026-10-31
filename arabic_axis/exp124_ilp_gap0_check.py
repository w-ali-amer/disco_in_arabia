import sys, json, random, numpy as np
ROOT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)); sys.path.insert(0, ROOT)
import exp124_spike as E
from scipy.optimize import milp, LinearConstraint, Bounds
d = json.load(open(ROOT + "/results_exp124.json"))
rng = random.Random(124)
bad = 0; maxdiff = 0; st = set(); i = 0
for R in (3, 4):
    for m in (6, 8, 10, 12, 14, 16):
        for s in range(50):
            inst = E.instance(rng, R, m); rec = d["records"][i]; i += 1
            assert (rec["R"], rec["m"], rec["seed"]) == (R, m, s)
            qp = E.f1_program(inst); n = qp.get_num_vars(); c = -qp.objective.linear.to_array()
            A, lo, hi = [], [], []
            for con in qp.linear_constraints:
                A.append(con.linear.to_array()); sn = con.sense.name
                lo.append(-np.inf if sn == "LE" else con.rhs); hi.append(con.rhs if sn in ("LE", "EQ") else np.inf)
            r = milp(c, constraints=[LinearConstraint(np.array(A), lo, hi)] if A else [], integrality=np.ones(n), bounds=Bounds(0, 1), options={"mip_rel_gap": 0})
            st.add(r.status); v = -r.fun; diff = abs(v - rec["F1"]["ilp_opt"]); maxdiff = max(maxdiff, diff); bad += diff > 1e-9
print("statuses", st, "instances differing", bad, "max |diff|", maxdiff)
