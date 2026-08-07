"""torch_sv_sim.py -- minimal differentiable statevector simulator in pure torch.

Built for exp40 (mini-Duneau calibration, doc 22 SS9 / SS3.3-3.4):
- n-qubit state as a torch tensor of shape (2,)*n (axis i == wire i).
- apply arbitrary 1q/2q/3q gates on given wires via tensordot; parameterized
  gates (RX, RZ, CRX) built from torch params so autodiff flows end-to-end.
- Sim-et-al. "Circuit 4" ansatz block (per layer: RX+RZ on each wire, then a
  CRX cascade), n layers, as a torch nn.Module exposing its full unitary.
- Duneau-style readout helper: P(queried wires == |0..0>) with every other
  wire (including follows-ancillas) traced out -- differentiable.
- Statevector norm is asserted preserved after every gate (CHECK_NORM).

No SPSA anywhere; exact gradients only.
"""

import math

import torch
from torch import nn

CDTYPE = torch.complex64
CHECK_NORM = True
NORM_TOL = 1e-3


def _c(x):
    """Real tensor -> complex dtype (differentiable cast)."""
    return x.to(CDTYPE)


def rx(theta):
    """RX(theta) = exp(-i theta X / 2)."""
    half = theta * 0.5
    c = _c(torch.cos(half))
    s = _c(torch.sin(half))
    i = torch.tensor(1j, dtype=CDTYPE)
    return torch.stack([torch.stack([c, -i * s]),
                        torch.stack([-i * s, c])])


def rz(theta):
    """RZ(theta) = diag(exp(-i theta/2), exp(+i theta/2))."""
    half = _c(theta * 0.5)
    i = torch.tensor(1j, dtype=CDTYPE)
    em = torch.exp(-i * half)
    ep = torch.exp(i * half)
    zero = torch.zeros((), dtype=CDTYPE)
    return torch.stack([torch.stack([em, zero]),
                        torch.stack([zero, ep])])


def crx(theta):
    """Controlled-RX; wire order [control, target] (control = first index)."""
    r = rx(theta)
    eye = torch.eye(2, dtype=CDTYPE)
    zero = torch.zeros((2, 2), dtype=CDTYPE)
    top = torch.cat([eye, zero], dim=1)
    bot = torch.cat([zero, r], dim=1)
    return torch.cat([top, bot], dim=0)


def euler_1q(a, b, c):
    """Rx-Rz-Rx Euler unitary: RX(c) @ RZ(b) @ RX(a) (RX(a) applied first)."""
    return rx(c) @ rz(b) @ rx(a)


def init_state(n):
    """|0...0> on n wires."""
    psi = torch.zeros((2,) * n, dtype=CDTYPE)
    psi[(0,) * n] = 1.0
    return psi


def state_norm(state):
    return torch.linalg.vector_norm(state)


def apply_gate(state, U, wires):
    """Apply a 2^k x 2^k unitary U to `wires` (tuple of k axis indices)."""
    k = len(wires)
    Ut = U.reshape((2,) * (2 * k))
    out = torch.tensordot(Ut, state, dims=(list(range(k, 2 * k)), list(wires)))
    out = torch.movedim(out, list(range(k)), list(wires))
    if CHECK_NORM:
        nrm = state_norm(out).item()
        assert abs(nrm - 1.0) < NORM_TOL, \
            "statevector norm not preserved: %.6f (wires=%s)" % (nrm, wires)
    return out


def prob_all_zero_on(state, wires):
    """P(wires == |0..0>) with ALL other wires traced out.

    = sum over all other-wire basis states of |amplitude|^2. Differentiable.
    """
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s.abs() ** 2).sum()


def mat_apply(M, U, wires, n):
    """Left-multiply the embedded gate: returns embed(U, wires) @ M.

    M is a (2^n, m) matrix whose columns are states; used to build the full
    unitary of a small circuit (n <= 3 here, so this is cheap).
    """
    m = M.shape[1]
    T = M.reshape((2,) * n + (m,))
    k = len(wires)
    Ut = U.reshape((2,) * (2 * k))
    T = torch.tensordot(Ut, T, dims=(list(range(k, 2 * k)), list(wires)))
    T = torch.movedim(T, list(range(k)), list(wires))
    return T.reshape(2 ** n, m)


class Circuit4(nn.Module):
    """Sim-et-al. Circuit 4 ansatz, `n_layers` layers on `n_qubits` wires.

    Per layer: RX(theta) then RZ(phi) on each wire, then a CRX cascade
    CRX(control=w+1, target=w) for w = n-2 .. 0.
    Parameters per layer: 2*n + (n-1). Init: uniform in [0, 2*pi).
    """

    def __init__(self, n_qubits, n_layers=3, generator=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        per_layer = 2 * n_qubits + (n_qubits - 1)
        init = torch.rand((n_layers, per_layer), generator=generator) * (2 * math.pi)
        self.theta = nn.Parameter(init)

    def matrix(self):
        n = self.n_qubits
        M = torch.eye(2 ** n, dtype=CDTYPE)
        for layer in range(self.n_layers):
            p = self.theta[layer]
            idx = 0
            for w in range(n):
                M = mat_apply(M, rx(p[idx]), (w,), n)
                idx += 1
            for w in range(n):
                M = mat_apply(M, rz(p[idx]), (w,), n)
                idx += 1
            for w in range(n - 2, -1, -1):
                M = mat_apply(M, crx(p[idx]), (w + 1, w), n)
                idx += 1
        return M


def _selftest():
    torch.manual_seed(0)
    # 1) Norm preservation through a random parameterized circuit on 4 wires.
    psi = init_state(4)
    params = torch.rand(6, requires_grad=True)
    psi = apply_gate(psi, euler_1q(params[0], params[1], params[2]), (0,))
    psi = apply_gate(psi, rx(params[3]), (2,))
    psi = apply_gate(psi, crx(params[4]), (0, 1))
    psi = apply_gate(psi, crx(params[5]), (2, 3))
    assert abs(state_norm(psi).item() - 1.0) < NORM_TOL

    # 2) Readout is a probability and grads flow (nonzero grad norm).
    p = prob_all_zero_on(psi, (0, 3))
    assert 0.0 <= p.item() <= 1.0 + 1e-6
    p.backward()
    gn = params.grad.norm().item()
    assert gn > 0.0, "gradient norm is zero"

    # 3) Circuit4 matrix is unitary and differentiable.
    g = torch.Generator().manual_seed(7)
    block = Circuit4(3, 3, generator=g)
    M = block.matrix()
    eye = torch.eye(8, dtype=CDTYPE)
    err = (M.conj().T @ M - eye).abs().max().item()
    assert err < 1e-4, "Circuit4 matrix not unitary: err=%g" % err
    psi = init_state(3)
    psi = apply_gate(psi, M, (0, 1, 2))
    loss = prob_all_zero_on(psi, (0,))
    loss.backward()
    assert block.theta.grad is not None and block.theta.grad.norm().item() > 0.0

    # 4) Physics: RX(pi) flips |0> -> |1| up to phase; CRX(pi) on |10> -> |11>.
    psi = init_state(1)
    psi = apply_gate(psi, rx(torch.tensor(math.pi)), (0,))
    assert abs(psi[1].abs().item() - 1.0) < 1e-5
    psi = init_state(2)
    psi = apply_gate(psi, rx(torch.tensor(math.pi)), (0,))       # control -> |1>
    psi = apply_gate(psi, crx(torch.tensor(math.pi)), (0, 1))    # flips target
    assert abs(psi[1, 1].abs().item() - 1.0) < 1e-5
    print("[torch_sv_sim] selftest OK (norms, grads, unitarity, gate physics)")


if __name__ == "__main__":
    _selftest()
