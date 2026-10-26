"""diag3_exp40.py -- isolate the remaining exp40 bottleneck.

A. walk/turn-only 2-actor stories (no follows): if the ansatz + question
   effects are right, this must hit 100% within a few epochs. Failure here
   => bug in the effects / single-qubit verbs, not in `follows`.
B. follows-inclusive 2-actor sanity, longer budget + higher lr: does it
   converge EVENTUALLY (optimization speed) or plateau (expressivity)?
C. same sanity with a 6-layer follows block: if depth unblocks it, the
   3-layer Circuit-4 `follows` is the expressivity bottleneck.
"""

import random

import torch

torch.set_num_threads(4)

import duneau_task_gen as taskgen
import exp40_mini_duneau as exp
import torch_sv_sim as sim


def make_walkturn_stories(rng, n, width=2):
    out = []
    want = True
    while len(out) < n:
        actors = rng.sample(taskgen.NAMES, width)
        events = [("walks", a, rng.choice(taskgen.DIRECTIONS)) for a in actors]
        for _ in range(rng.randint(width, 2 * width)):
            events.append(("turns", rng.choice(actors)))
        dirs = taskgen.simulate(events)
        pairs = [(a, b) for i, a in enumerate(actors) for b in actors[i + 1:]]
        pool = [(a, b) for a, b in pairs if (dirs[a] == dirs[b]) == want]
        if not pool:
            continue
        x, y = rng.choice(pool)
        out.append({"width": width, "actors": actors,
                    "events": [list(e) for e in events],
                    "question": [x, y], "answer": "yes" if want else "no",
                    "n_ancilla": 0})
        want = not want
    return out


def run(train, valid, seed, epochs, lr, tag, follows_layers=None):
    torch.manual_seed(seed)
    model = exp.Exp40Model(seed)
    if follows_layers is not None:
        g = torch.Generator().manual_seed(seed + 777)
        model.follows = sim.Circuit4(3, follows_layers, generator=g)
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
    rng = random.Random(99)
    data = taskgen.generate_dataset(40)
    w2 = [s for s in data["train"] if s["width"] == 2]
    w2v = [s for s in data["validA"] if s["width"] == 2]

    print("== A: walk/turn-only 2-actor (no follows) ==", flush=True)
    wt = make_walkturn_stories(rng, 40)
    wtv = make_walkturn_stories(rng, 12)
    run(wt, wtv, seed=0, epochs=15, lr=0.005, tag="A lr.005")

    print("== B: follows-inclusive 2-actor, longer/higher lr ==", flush=True)
    run(w2, w2v, seed=0, epochs=40, lr=0.02, tag="B lr.02")

    print("== C: follows block with 6 layers ==", flush=True)
    run(w2, w2v, seed=0, epochs=40, lr=0.02, tag="C 6L lr.02", follows_layers=6)


if __name__ == "__main__":
    main()
