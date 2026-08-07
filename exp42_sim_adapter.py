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
