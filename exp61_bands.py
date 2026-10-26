# -*- coding: utf-8 -*-
"""exp61_bands.py -- doc 47 SS5.1: THE shared band geometry.

The ceiling computation and the training runs MUST import this same module, or
they silently measure different geometries (the doc-42 failure class). The
frozen ceiling table is asserted against a live recomputation at import.
"""
import itertools, math
import numpy as np
from scipy.optimize import linprog

PI = math.pi
W = 0.05                                   # band half-width, frozen (doc 47 SS6)

# doc 47 SS6 -- FROZEN configurations: id -> (m bands per class, spacing)
CONFIGS = {"M1": (1, 0.40), "M2": (2, 0.40), "M2w": (2, 0.80), "M3": (3, 0.40)}

# doc 47 SS3.3 -- exact ceilings, derived by LP over subsets of band-pairs
FROZEN_CEILINGS = {("M1", 1): 1.0, ("M1", 2): 1.0,
                   ("M2", 1): 0.75, ("M2", 2): 1.0,
                   ("M2w", 1): 0.75, ("M2w", 2): 1.0,
                   ("M3", 1): 6 / 9, ("M3", 2): 8 / 9}


# doc 47 SS17 ERRATUM: FROZEN_CEILINGS above required both twins of a band-pair
# to be won together (antisymmetric part), which is SUFFICIENT but not NECESSARY
# -- per-twin partial credit is possible with a general C. Those values are
# therefore LOWER BOUNDS. These are the corrected ceilings, and they are the
# ones every verdict is read against.
CORRECTED_CEILINGS = {("M1", 1): 1.0, ("M1", 2): 1.0,
                      ("M2", 1): 0.875, ("M2", 2): 1.0,
                      ("M2w", 1): 0.875, ("M2w", 2): 1.0,
                      ("M3", 1): 7 / 9, ("M3", 2): 17 / 18}


def true_ceiling(cfg, K):
    """doc 47 SS17: MAX-FS over PER-TWIN constraints in a GENERAL C."""
    X = [_u(c, K) for c in bearer_bands(cfg)]
    Y = [_u(c, K) for c in nonbearer_bands(cfg)]
    cons = []
    for x in X:
        for y in Y:
            cons.append(np.outer(x, y).ravel())        # x^T C y > 0
            cons.append(-np.outer(y, x).ravel())       # y^T C x < 0
    n, d = len(cons), len(cons[0])
    for keep in range(n, 0, -1):
        for sub in itertools.combinations(range(n), keep):
            A = np.array([-cons[i] for i in sub])
            if linprog(np.zeros(d), A_ub=A, b_ub=-np.ones(keep),
                       bounds=[(-1e4, 1e4)] * d, method="highs").status == 0:
                return keep / n
    return 0.0


def verify_corrected_table():
    bad = [(c, K, w, true_ceiling(c, K)) for (c, K), w in CORRECTED_CEILINGS.items()
           if abs(true_ceiling(c, K) - w) > 1e-9]
    assert not bad, "CORRECTED CEILING MISMATCH: %s" % bad
    return True


def band_centres(cfg):
    m, s = CONFIGS[cfg]
    return [PI + (i - (2 * m - 1) / 2) * s for i in range(2 * m)]


def bearer_bands(cfg):
    return [c for i, c in enumerate(band_centres(cfg)) if i % 2 == 0]


def nonbearer_bands(cfg):
    return [c for i, c in enumerate(band_centres(cfg)) if i % 2 == 1]


def assert_geometry(cfg):
    """doc 47 SS6: wrap-around clearance, asserted not assumed."""
    cs = band_centres(cfg)
    lo, hi = min(cs) - W, max(cs) + W
    assert 0 < lo and hi < 2 * PI, "band wraps 0/2pi: %s [%.3f,%.3f]" % (cfg, lo, hi)
    m, s = CONFIGS[cfg]
    assert s > 2 * W, "bands overlap: s=%.3f w=%.3f" % (s, W)
    return lo, hi


def assign_bands(cfg, bearers, nonbearers, train_inv, held_inv, seed=61000):
    """Round-robin nouns over the m bands of their class at a REGISTERED seed.
    doc 47 SS5: every band must hold >=1 train AND >=1 held-out noun, or
    generalisation confounds with band-extrapolation."""
    import random
    m, _s = CONFIGS[cfg]
    rng = random.Random(seed)
    out = {}
    for pool, centres in ((sorted(bearers), bearer_bands(cfg)),
                          (sorted(nonbearers), nonbearer_bands(cfg))):
        p = list(pool)
        rng.shuffle(p)
        for k, n in enumerate(p):
            out[n] = centres[k % m]
    for centres in (bearer_bands(cfg), nonbearer_bands(cfg)):
        for c in centres:
            tr = [n for n in train_inv if out.get(n) == c]
            he = [n for n in held_inv if out.get(n) == c]
            assert tr and he, ("band %.3f in %s lacks train(%d)/held-out(%d) "
                              "coverage" % (c, cfg, len(tr), len(he)))
    return out


def build_angles(cfg, seed, bearers, nonbearers, train_inv, held_inv,
                 flat=False, assign_seed=61000):
    """THE angle map. flat=True collapses every band to pi (category absent)."""
    import random
    assert_geometry(cfg)
    band = assign_bands(cfg, bearers, nonbearers, train_inv, held_inv, assign_seed)
    rng = random.Random(61500 + seed)
    return {n: (PI if flat else band[n]) + rng.uniform(-W, W) for n in band}


# ------------------------------------------------------------------ ceiling
def _u(t, K):
    v = [1.0]
    for k in range(1, K + 1):
        v += [math.cos(k * t), math.sin(k * t)]
    return np.array(v)


def _feasible(pairs):
    """Exists an antisymmetric A with x^T A y >= 1 for every listed pair?"""
    if not pairs:
        return True
    d = pairs[0][0].shape[0]
    idx = [(a, b) for a in range(d) for b in range(a + 1, d)]
    A = [[-(x[a] * y[b] - x[b] * y[a]) for a, b in idx] for x, y in pairs]
    return linprog(np.zeros(len(idx)), A_ub=np.array(A),
                   b_ub=-np.ones(len(pairs)),
                   bounds=[(-1e4, 1e4)] * len(idx), method="highs").status == 0


def exact_ceiling(cfg, K):
    """doc 47 SS3.2: max over antisymmetric A of the fraction of band-pairs with
    x^T A y > 0. Exact by LP over subsets."""
    X = [_u(c, K) for c in bearer_bands(cfg)]
    Y = [_u(c, K) for c in nonbearer_bands(cfg)]
    pairs = [(x, y) for x in X for y in Y]
    n = len(pairs)
    for keep in range(n, 0, -1):
        for sub in itertools.combinations(range(n), keep):
            if _feasible([pairs[i] for i in sub]):
                return keep / n
    return 0.0


def verify_frozen_table():
    bad = []
    for (cfg, K), want in FROZEN_CEILINGS.items():
        got = exact_ceiling(cfg, K)
        if abs(got - want) > 1e-9:
            bad.append((cfg, K, want, got))
    assert not bad, "FROZEN CEILING MISMATCH: %s" % bad
    return True


if __name__ == "__main__":
    for cfg in CONFIGS:
        lo, hi = assert_geometry(cfg)
        print("%-4s m=%d s=%.2f  span [%.3f, %.3f]  K=1 %.4f  K=2 %.4f"
              % (cfg, CONFIGS[cfg][0], CONFIGS[cfg][1], lo, hi,
                 exact_ceiling(cfg, 1), exact_ceiling(cfg, 2)))
    print("frozen table verified:", verify_frozen_table())
