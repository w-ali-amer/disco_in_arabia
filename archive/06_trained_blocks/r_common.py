# -*- coding: utf-8 -*-
"""r_common.py -- shared plumbing for the CLEAN-DATA RE-DECISION (doc 25 SSR,
commit 6d5c6a4; pointer in doc 28). New file; no existing file is edited.

What lives here (and nothing else):
  * the dataset pins for the regenerated L2r rung and family-r (filled from
    r_datagen's output; every r_ runner refuses to run on any other bytes);
  * the registered readout policy (doc 25 SSR.4 "Readout"): smooth
    |z|^2 = re^2 + im^2 in every quantum arm, installed by monkeypatching
    torch_sv_sim.prob_all_zero_on AND the exp42_sim_adapter re-export (the
    only two binding sites through which A3/A4/A5 read out), recorded with an
    A0-style forward anchor against the original abs()**2 form;
  * Rule-3-safe path helpers (refuse to overwrite, never suffix-shop).
"""

import hashlib
import inspect
import json
import os

import torch

# ------------------------------------------------------------------ pins
L2R_DATA = "stories_exp43b_L2r.json"
FAMR_DATA = "stories_exp43a_familyr.json"
L2R_SEED = 4312000                       # doc 25 SSR.1 (x1000 precedent)
FAMR_SEED = 4301000
# canonical items sha256 (json.dumps(items, ensure_ascii=False,
# sort_keys=True)), pinned from r_datagen_record.json after generation.
L2R_SHA256 = "f16749f0cadfff929bc62b6064417e7e52ce77b8b1da0d89971d1fe692b15e25"
FAMR_SHA256 = "2e6650d6a2ea8e11c8eaee3a0b4d7c275c985203dbb2318675e3fb78abc4535a"
# the UNCHANGED L1 rung (registered L1 reuse, doc 25 SSR.4(b))
L1_DATA = "stories_exp43b_L1.json"
L1_SHA256 = "88cd70ad530356911c17f18d0b42de3edf3c956324513a1455c9548aaef15534"
EXP45B_ORIG_RESULTS = "results_exp45b_phase1.json"
EXP45B_ORIG_CKPT = "exp45b_ckpt"

REGISTRATION = ("qnlp_private_docs/25_exp43b_hop_split_rerun.md SSR "
                "(CLEAN-DATA RE-DECISION, commit 6d5c6a4) + doc 28 pointer")


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_pinned(path, pin):
    """Load a dataset, recompute its canonical sha, require meta == pin."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sha = canonical_sha256(d["items"])
    if sha != d["meta"]["dataset_sha256"] or sha != pin:
        raise RuntimeError("dataset provenance FAILED for %s: recomputed=%s "
                           "meta=%s pin=%s" % (path, sha,
                                               d["meta"]["dataset_sha256"], pin))
    return d["items"], d["meta"], sha


def source_sha256(obj):
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


# ------------------------------------------------------------ Rule 3 paths
def rule3(path):
    """Return `path` only if nothing exists there (research rule 3)."""
    if os.path.exists(path):
        raise FileExistsError("Rule 3: refusing to overwrite %s" % path)
    return path


def make_fresh_results_path(mapping, outdir="."):
    """Replacement for exp42_run.fresh_results_path inside a wrapped runner:
    maps the original base name to the NEW registered name and refuses to
    overwrite (no b/c/d suffix shopping)."""
    def fresh_results_path(base):
        if base not in mapping:
            raise RuntimeError("unmapped results base %r (wrapper refuses to "
                               "write under an original name)" % base)
        return rule3(os.path.join(outdir, mapping[base] + ".json"))
    return fresh_results_path


def inject_provenance(path, block, key="r_wrapper"):
    """Add the wrapper's provenance block to a results file the wrapped runner
    has JUST created in this process (never to a pre-existing file)."""
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    assert key not in out
    out[key] = block
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------- readout policy
def _p0_smooth(state, wires):
    """P(wires == |0..0>): identical maths to torch_sv_sim.prob_all_zero_on,
    |z|^2 computed as re^2 + im^2 (exp90_models._p0; doc 83 SS11.3): complex64
    abs() backward is NaN at subnormal amplitudes; re^2+im^2 is smooth."""
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s.real ** 2 + s.imag ** 2).sum()


_PATCH_RECORD = None


def patch_readout():
    """Install the registered smooth readout at every binding site; return a
    record (idempotent)."""
    global _PATCH_RECORD
    if _PATCH_RECORD is not None:
        return _PATCH_RECORD
    import torch_sv_sim
    import exp42_sim_adapter
    orig = torch_sv_sim.prob_all_zero_on
    assert exp42_sim_adapter.prob_all_zero_on is orig, \
        "adapter binding drifted from torch_sv_sim"
    orig_src_sha = source_sha256(orig)

    # A0 anchor: forward agreement on random normal-range states
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for n in (4, 6, 8, 10, 12):
        for _ in range(20):
            re = torch.randn((2,) * n, generator=g)
            im = torch.randn((2,) * n, generator=g)
            psi = torch.complex(re, im).to(torch_sv_sim.CDTYPE)
            psi = psi / torch.sqrt((psi.abs() ** 2).sum())
            wires = tuple(range(0, min(4, n)))
            d = abs(orig(psi, wires).item() - _p0_smooth(psi, wires).item())
            worst = max(worst, d)
    # the bug the policy removes: gradient at a subnormal amplitude
    z = torch.tensor([1.1e-39 + 0j, 1.0 + 0j],
                     dtype=torch_sv_sim.CDTYPE).reshape(2).requires_grad_(True)
    (z.abs() ** 2).sum().backward()
    orig_grad_finite = bool(torch.isfinite(torch.view_as_real(z.grad)).all())
    z2 = torch.tensor([1.1e-39 + 0j, 1.0 + 0j],
                      dtype=torch_sv_sim.CDTYPE).reshape(2).requires_grad_(True)
    (z2.real ** 2 + z2.imag ** 2).sum().backward()
    smooth_grad_finite = bool(torch.isfinite(torch.view_as_real(z2.grad)).all())

    torch_sv_sim.prob_all_zero_on = _p0_smooth
    exp42_sim_adapter.prob_all_zero_on = _p0_smooth
    _PATCH_RECORD = {
        "policy": "doc 25 SSR.4: smooth |z|^2 = re^2 + im^2 in every quantum "
                  "arm; isfinite asserts kept (and added where missing); a "
                  "non-finite aborts that restart/run and is recorded",
        "patched_sites": ["torch_sv_sim.prob_all_zero_on",
                          "exp42_sim_adapter.prob_all_zero_on"],
        "original_source_sha256": orig_src_sha,
        "replacement_source_sha256": source_sha256(_p0_smooth),
        "forward_anchor_max_abs_delta": worst,
        "subnormal_grad_finite_original_abs2": orig_grad_finite,
        "subnormal_grad_finite_smooth": smooth_grad_finite,
    }
    return _PATCH_RECORD


def assert_patched():
    import torch_sv_sim
    import exp42_sim_adapter
    assert torch_sv_sim.prob_all_zero_on is _p0_smooth
    assert exp42_sim_adapter.prob_all_zero_on is _p0_smooth
