# -*- coding: utf-8 -*-
"""exp62_bands.py -- band assignment for the THREE-WAY inventory (doc 48 SS8.6).

exp61_bands.assign_bands round-robins over the GLOBAL noun list, which cannot
guarantee that each inventory covers every band -- with 4 test bearers over 2
bands the global order left one band with zero test nouns. Coverage is required
(doc 47 SS5: otherwise generalisation confounds with band-extrapolation), so
assignment here is STRATIFIED WITHIN EACH INVENTORY.

exp61_bands.assign_bands is deliberately NOT modified: exp61's results depend on
it bit-for-bit.
"""
import math, random
from exp61_bands import (CONFIGS, W, PI, band_centres, bearer_bands,
                         nonbearer_bands, assert_geometry, FROZEN_CEILINGS,
                         CORRECTED_CEILINGS, exact_ceiling, true_ceiling)


def assign_bands_3way(cfg, bearers, nonbearers, inventories, seed=62000):
    """Round-robin WITHIN each inventory, so every inventory covers every band."""
    m, _s = CONFIGS[cfg]
    rng = random.Random(seed)
    out = {}
    for pool, centres in ((set(bearers), bearer_bands(cfg)),
                          (set(nonbearers), nonbearer_bands(cfg))):
        for inv in inventories:
            p = sorted(set(inv) & pool)
            rng.shuffle(p)
            for k, n in enumerate(p):
                out[n] = centres[k % m]
    for centres in (bearer_bands(cfg), nonbearer_bands(cfg)):
        for c in centres:
            for inv in inventories:
                hit = [n for n in inv if out.get(n) == c]
                assert hit, ("band %.3f in %s has NO noun from an inventory "
                             "(size %d)" % (c, cfg, len(inv)))
    return out


def build_angles_3way(cfg, seed, bearers, nonbearers, inventories,
                      flat=False, assign_seed=62000):
    assert_geometry(cfg)
    band = assign_bands_3way(cfg, bearers, nonbearers, inventories, assign_seed)
    rng = random.Random(62500 + seed)
    return {n: (PI if flat else band[n]) + rng.uniform(-W, W) for n in band}
