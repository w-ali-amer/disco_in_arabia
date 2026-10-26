"""diag2_exp40.py -- verify the exp40b harness fixes before the 5-seed rerun.

1. Simulator wire-order unit check: apply_gate on arbitrarily ordered wires
   must equal the explicit permuted matrix (rules out a sim bug for good).
2. 2-actor sanity on generator v1.1 (walks introduction-only), batch 1,
   lr 0.005, log-p loss: must reach 100% within a few epochs.
3. Short full-widths probe (1 seed, 6 epochs) to see the fixed trajectory.
"""

import random

import torch
import torch.nn.functional as F

torch.set_num_threads(4)

import duneau_task_gen as taskgen
import exp40_mini_duneau as exp
import torch_sv_sim as sim


def wire_order_check():
    torch.manual_seed(3)
    # Random 2q unitary from QR decomposition.
    A = torch.randn(4, 4, dtype=torch.complex64)
    Q, _ = torch.linalg.qr(A)
    psi = torch.randn(2, 2, 2, dtype=torch.complex64)
    psi = psi / torch.linalg.vector_norm(psi)
    sim_ck, sim.CHECK_NORM = sim.CHECK_NORM, False  # random state, skip norm-1 assert
    try:
        out = sim.apply_gate(psi, Q, (2, 0))
        # Reference: explicit index arithmetic over the 8 basis states.
        ref = torch.zeros_like(psi)
        for b0 in range(2):
            for b1 in range(2):
                for b2 in range(2):
                    acc = 0
                    for a0 in range(2):
                        for a2 in range(2):
                            # gate acts on (wire2, wire0): row=(b2,b0), col=(a2,a0)
                            acc = acc + Q[b2 * 2 + b0, a2 * 2 + a0] * psi[a0, b1, a2]
                    ref[b0, b1, b2] = acc
        err = (out - ref).abs().max().item()
        assert err < 1e-5, "wire-order mismatch: %g" % err
        print("[diag2] wire-order check OK (max err %.2e)" % err, flush=True)
    finally:
        sim.CHECK_NORM = sim_ck


def run_train(train, valid, seed, epochs, lr=0.005, batch=1, tag=""):
    torch.manual_seed(seed)
    model = exp.Exp40Model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        correct = 0
        for start in range(0, len(order), batch):
            chunk = [train[i] for i in order[start:start + batch]]
            opt.zero_grad()
            loss = 0.0
            for s in chunk:
                l, pred = model.story_loss(s)
                loss = loss + l
                if pred == exp.LABELS[s["answer"]]:
                    correct += 1
            (loss / len(chunk)).backward()
            opt.step()
        tr = correct / len(train)
        va = exp.evaluate(model, valid)
        print("  [%s] seed=%d epoch %2d  train(run)=%.3f valid=%.3f"
              % (tag, seed, epoch, tr, va), flush=True)
        if tr >= 1.0 and va >= 1.0:
            print("  [%s] converged at epoch %d" % (tag, epoch), flush=True)
            return epoch
    return None


def main():
    wire_order_check()
    data = taskgen.generate_dataset(40)
    print("[diag2] generator v%s  sha256=%s"
          % (taskgen.GENERATOR_VERSION, data["meta"]["sha256"][:16]), flush=True)

    w2 = [s for s in data["train"] if s["width"] == 2]
    w2v = [s for s in data["validA"] if s["width"] == 2]
    print("[diag2] 2-actor sanity (%d train / %d valid), batch 1, lr 0.005, log-p"
          % (len(w2), len(w2v)), flush=True)
    ep = run_train(w2, w2v, seed=0, epochs=10, tag="sanity")
    print("[diag2] 2-actor sanity: %s"
          % (("PASS (epoch %d)" % ep) if ep else "FAIL (10 epochs)"), flush=True)

    print("[diag2] full-widths probe: seed 0, 6 epochs, batch 1", flush=True)
    run_train(data["train"], data["validA"], seed=0, epochs=6, tag="probe")


if __name__ == "__main__":
    main()
