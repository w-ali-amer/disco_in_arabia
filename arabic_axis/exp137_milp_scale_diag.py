import sys; sys.path.insert(0, ".")
import exp137_e2 as E2, exp137_run as R137, exp110_run as R110, exp116_run as E116
p = R137.priors()["E-6"]; R110.uniform_rows = lambda K: E2.rows(K, list(p))
orig = E116.milp; S = float(sys.argv[1])
E116.milp = lambda c, **kw: orig(c * S, **kw)
r = E116.solve(("diag", 3, (0, 1, 2, 2), 3600, True))
print("scale", S, "value", r["value"] / S, "proved", r["proved"], "wall", int(r["wall"]), "enum", 0.7776176144482867)
