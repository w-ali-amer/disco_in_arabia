"""diag4_exp40.py -- test the follows wire-order hypothesis.

Circuit-4's CRX cascade couples only adjacent wires. The follows channel
needs follower<->followee coupling (copy in) AND follower<->ancilla coupling
(dump the follower's old state; required by unitarity). With wire order
(follower, followee, ancilla) the follower-ancilla coupling is absent.
Test order (followee, follower, ancilla) -- follower in the middle -- by
monkeypatching the wire tuple used when applying the follows unitary.

D. follows-inclusive 2-actor sanity, 3-layer follows, follower-middle order,
   lr 0.005 and lr 0.02.
E. control: same but full-widths probe (seed 0, 8 epochs, lr 0.005).
"""

import random

import torch

torch.set_num_threads(4)

import duneau_task_gen as taskgen
import exp40_mini_duneau as exp
import torch_sv_sim as sim


class MiddleFollowerModel(exp.Exp40Model):
    """Exp40Model with follows applied on (followee, follower, ancilla)."""

    def story_values(self, story):
        actors = story["actors"]
        K = len(actors)
        wire = {a: i for i, a in enumerate(actors)}
        n = K + story["n_ancilla"]
        U_person = sim.euler_1q(self.person[0], self.person[1], self.person[2])
        U_north = self._verb("walks_north")
        U_south = self._verb("walks_south")
        U_turn = self._verb("turns")
        U_follow = self.follows.matrix()
        psi = sim.init_state(n)
        for i in range(K):
            psi = sim.apply_gate(psi, U_person, (i,))
        next_anc = K
        for ev in story["events"]:
            kind = ev[0]
            if kind == "walks":
                _, a, d = ev
                psi = sim.apply_gate(psi, U_north if d == "north" else U_south,
                                     (wire[a],))
            elif kind == "turns":
                psi = sim.apply_gate(psi, U_turn, (wire[ev[1]],))
            elif kind in ("follows", "opposite"):
                _, a, b = ev
                # follower in the MIDDLE wire slot
                psi = sim.apply_gate(psi, U_follow,
                                     (wire[b], wire[a], next_anc))
                next_anc += 1
                if kind == "opposite":
                    psi = sim.apply_gate(psi, U_turn, (wire[a],))
        x, y = story["question"]
        qw = (wire[x], wire[y])
        vals = []
        for effect in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, effect.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, qw))
        return torch.stack(vals)


def run(model_cls, train, valid, seed, epochs, lr, tag):
    torch.manual_seed(seed)
    model = model_cls(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    best = (0.0, 0.0)
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        correct = 0
        for i in order:
            opt.zero_grad()
            loss, pred = model.story_loss(train[i])
            loss.backward()
            opt.step()
            if pred == exp.LABELS[train[i]["answer"]]:
                correct += 1
        tr = correct / len(train)
        va = exp.evaluate(model, valid)
        best = max(best, (tr, va))
        if epoch % 5 == 0 or (tr >= 1.0 and va >= 1.0) or epoch <= 3:
            print("  [%s] epoch %2d  train(run)=%.3f valid=%.3f"
                  % (tag, epoch, tr, va), flush=True)
        if tr >= 1.0 and va >= 1.0:
            print("  [%s] CONVERGED at epoch %d" % (tag, epoch), flush=True)
            return epoch
    print("  [%s] NOT converged in %d epochs (best train=%.3f valid=%.3f)"
          % (tag, epochs, best[0], best[1]), flush=True)
    return None


def main():
    data = taskgen.generate_dataset(40)
    w2 = [s for s in data["train"] if s["width"] == 2]
    w2v = [s for s in data["validA"] if s["width"] == 2]

    print("== D: follower-middle order, 2-actor sanity, 3 layers ==", flush=True)
    run(MiddleFollowerModel, w2, w2v, 0, 40, 0.005, "D lr.005")
    run(MiddleFollowerModel, w2, w2v, 0, 40, 0.02, "D lr.02")

    print("== E: follower-middle order, full-widths probe ==", flush=True)
    run(MiddleFollowerModel, data["train"], data["validA"], 0, 8, 0.005,
        "E lr.005")


if __name__ == "__main__":
    main()
