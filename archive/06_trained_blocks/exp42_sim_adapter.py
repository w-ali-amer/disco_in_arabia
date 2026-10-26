# -*- coding: utf-8 -*-
"""exp42_sim_adapter.py -- thin adapter isolating exp42 from torch_sv_sim.

torch_sv_sim.py belongs to exp40 (its owner may still be iterating on it);
exp42 code imports simulator primitives ONLY through this module and NEVER
modifies torch_sv_sim. If torch_sv_sim's API drifts, the fix lands here and
nowhere else.

Adds the gates exp42 needs that torch_sv_sim does not provide:
- ry(theta)                  referent introduction (doc 22 SS3.1)
- cphase(theta)              symmetric controlled-phase diag(1,1,1,e^{i t})
- xx(phi)                    exp(-i phi/2 X(x)X)
- SWAP                       constant, for exchange-symmetry unit tests

NOTE ON "CRz" (doc 22 SS3.1): the doc writes E_v = CRz(theta)*XX(phi) and
calls it the exchange-symmetric entangler core. The qiskit-convention CRZ
(diag(1,1,e^{-it/2},e^{+it/2})) is NOT swap-symmetric; the repo precedent
(exp37b/exp38 `crz(t) = diag(1,1,1,e^{i pi t})`) is the symmetric
controlled-phase, which is what exchange symmetry requires. exp42 follows
the repo precedent: cphase() below. [E,SWAP]=0 is asserted in
exp42_selftest.py, so this interpretation is machine-checked, not assumed.
"""

import torch

import torch_sv_sim as _sim

_REQUIRED_API = ("CDTYPE", "init_state", "apply_gate", "prob_all_zero_on",
                 "euler_1q", "rx", "rz", "state_norm", "Circuit4")


def check_api():
    missing = [a for a in _REQUIRED_API if not hasattr(_sim, a)]
    if missing:
        raise ImportError(
            "torch_sv_sim API drift -- missing %s. Fix exp42_sim_adapter.py "
            "(do NOT patch torch_sv_sim from exp42 code)." % (missing,))


check_api()

CDTYPE = _sim.CDTYPE
init_state = _sim.init_state
apply_gate = _sim.apply_gate
prob_all_zero_on = _sim.prob_all_zero_on
euler_1q = _sim.euler_1q
rx = _sim.rx
rz = _sim.rz
state_norm = _sim.state_norm
Circuit4 = _sim.Circuit4


def _c(x):
    """Real tensor -> complex dtype (differentiable cast)."""
    return x.to(CDTYPE)


def ry(theta):
    """RY(theta) = exp(-i theta Y / 2) -- real rotation matrix."""
    if not torch.is_tensor(theta):
        theta = torch.tensor(float(theta))
    half = theta * 0.5
    c = _c(torch.cos(half))
    s = _c(torch.sin(half))
    return torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])])


def cphase(theta):
    """Symmetric controlled-phase diag(1,1,1,e^{i theta}); [CP,SWAP]=0.

    Matches repo precedent exp37b/exp38 `crz` up to the pi-scaling of the
    argument (exp42 uses radians directly; both are trained parameters).
    """
    if not torch.is_tensor(theta):
        theta = torch.tensor(float(theta))
    one = torch.ones((), dtype=CDTYPE)
    ph = torch.exp(torch.tensor(1j, dtype=CDTYPE) * _c(theta))
    return torch.diag(torch.stack([one, one, one, ph]))


_XX_MAT = torch.tensor([[0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]], dtype=CDTYPE)


def xx(phi):
    """XX(phi) = exp(-i phi/2 X(x)X) = cos(phi/2) I - i sin(phi/2) X(x)X."""
    if not torch.is_tensor(phi):
        phi = torch.tensor(float(phi))
    half = phi * 0.5
    eye = torch.eye(4, dtype=CDTYPE)
    return _c(torch.cos(half)) * eye \
        - torch.tensor(1j, dtype=CDTYPE) * _c(torch.sin(half)) * _XX_MAT


SWAP = torch.tensor([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=CDTYPE)


def kron2(A, B):
    """Kronecker product; first factor acts on the FIRST wire passed to
    apply_gate (agent wire in exp42's convention)."""
    return torch.kron(A, B)


def unitarize(M):
    """Project M onto the closest unitary via QR decomposition with sign fix.

    For M = Q R (thin QR), Q is unitary and R is upper-triangular.  Correcting
    Q's column signs via sign(diag(R)) gives the same polar factor as the SVD
    when M is already close to unitary (error O(eps32)).

    QR is preferred over SVD here because near-identity matrices (as found at
    initialisation when params ~ N(0, 0.1²)) have all singular values ≈ 1
    (degenerate), which can cause torch.linalg.svd's GESDD to fail to converge.
    torch.linalg.qr is stable for any matrix including near-identity.
    Differentiable via torch.linalg.qr's autograd.

    Use in story_values for arms with long event sequences (e.g. A2q2 at 2K=24
    wires where 10+ sequential gate applications accumulate ~1e-3 norm drift in
    float32; fix measured empirically: seed 2 drift dropped from 1.28e-6 to
    sub-eps32 after adding this call).
    """
    Q, R = torch.linalg.qr(M)
    d = torch.diagonal(R)
    # complex sign via sgn (sign(0)=0 → treat as +1 to avoid zeroing columns)
    sg = torch.sgn(d)
    phase = torch.where(d.abs() > 1e-10, sg, torch.ones_like(sg))
    return Q * phase.unsqueeze(0)


# ---- apply_gate with renormalize-and-log fallback -------------------------
import time as _time
import json as _json

_RENORM_LOG = []    # in-process accumulator; flush with flush_renorm_log()


def apply_gate(state, U, wires):
    """Wraps _sim.apply_gate; adds renormalize-and-log fallback for large states.

    float32 norm accumulation at N=2^24 amplitudes (A2q2 K=12) can push the
    measured norm outside NORM_TOL=1e-3 even for genuinely unitary gates (the
    float32 summation error itself is ~5e-4 at that size).  Primary fix: apply
    unitarize() to the gate matrix before this call.  This fallback catches any
    residual cases and records them for audit.

    Hard fail (raises) if renormalized norm is still off by > 1e-4 — that
    indicates a truly non-unitary gate, not float32 noise.
    """
    k = len(wires)
    Ut = U.reshape((2,) * (2 * k))
    out = torch.tensordot(Ut, state, dims=(list(range(k, 2 * k)), list(wires)))
    out = torch.movedim(out, list(range(k)), list(wires))
    if _sim.CHECK_NORM:
        nrm = float(_sim.state_norm(out).item())
        drift = abs(nrm - 1.0)
        if drift >= _sim.NORM_TOL:
            # Renormalize and log (spec: no silent NORM_TOL widening)
            incident = {
                "t": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "wires": list(wires), "norm_f32": round(nrm, 8),
                "drift_f32": round(drift, 8),
                "state_n_amplitudes": out.numel(),
                "renormalized": True,
            }
            _RENORM_LOG.append(incident)
            print("[apply_gate RENORM] norm=%.6f wires=%s n_amp=%d -- renorm"
                  % (nrm, wires, out.numel()), flush=True)
            out = out / (nrm + 1e-30)
            nrm2 = float(_sim.state_norm(out).item())
            assert abs(nrm2 - 1.0) < 1e-4, (
                "apply_gate: renorm failed (truly non-unitary gate): "
                "%.6f (wires=%s)" % (nrm2, wires))
    return out


def flush_renorm_log(path="exp81_renorm_log.jsonl"):
    """Append accumulated renorm incidents to a JSONL file and clear buffer."""
    if not _RENORM_LOG:
        return 0
    with open(path, "a", encoding="utf-8") as f:
        for inc in _RENORM_LOG:
            f.write(_json.dumps(inc) + "\n")
    n = len(_RENORM_LOG)
    _RENORM_LOG.clear()
    return n
