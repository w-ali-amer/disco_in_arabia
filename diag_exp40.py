"""diag_exp40.py -- harness diagnosis for the exp40 calibration failure.

Priority order (coordinator):
1. Gradient scale at the readout: raw p in [0,1] as softmax logits -> tiny
   logit gap, near-vanishing CE gradients. Log mean |p_pos - p_neg| and CE
   grad norms for raw-p logits vs log-p logits (log-p softmax == normalized
   p_pos/(p_pos+p_neg) with NLL -- same argmax prediction rule).
2. Config parity: batch 1, lr 0.005, plain Adam.
5. Sanity: 2-actor-only subtask should hit 100% within a few epochs if the
   circuits (Circuit-4 direction, question effect) are right.

Diagnostic only -- does not touch results_exp40.json or the running job.
"""

import random

import torch
import torch.nn.functional as F

torch.set_num_threads(2)  # be polite to the still-running exp40 job

import duneau_task_gen as taskgen
import exp40_mini_duneau as exp


def logits_of(vals, mode):
    if mode == "raw":
        return vals
    if mode == "logp":
        return torch.log(vals + 1e-12)
    raise ValueError(mode)


def diag_gradient_scale(data):
    print("== [1] readout gradient scale ==", flush=True)
    stories = data["train"][:20]
    for mode in ("raw", "logp"):
        for seed in (0, 1):
            torch.manual_seed(seed)
            model = exp.Exp40Model(seed)
            gaps, gns, losses = [], [], []
            for s in stories:
                vals = model.story_values(s)
                gaps.append(abs(vals[0].item() - vals[1].item()))
                lo = logits_of(vals, mode)
                loss = F.cross_entropy(
                    lo.unsqueeze(0),
                    torch.tensor([exp.LABELS[s["answer"]]]))
                model.zero_grad()
                loss.backward()
                gns.append(exp.grad_global_norm(model))
                losses.append(loss.item())
            print("  mode=%-4s seed=%d  mean|p_pos-p_neg|=%.4f  "
                  "mean CE grad norm=%.3e  mean loss=%.4f"
                  % (mode, seed, sum(gaps) / len(gaps),
                     sum(gns) / len(gns), sum(losses) / len(losses)),
                  flush=True)


def train_quick(train, valid, mode, seed, epochs, lr=0.005, batch=1):
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
                vals = model.story_values(s)
                lo = logits_of(vals, mode)
                loss = loss + F.cross_entropy(
                    lo.unsqueeze(0),
                    torch.tensor([exp.LABELS[s["answer"]]]))
                if int(torch.argmax(vals).item()) == exp.LABELS[s["answer"]]:
                    correct += 1
            (loss / len(chunk)).backward()
            opt.step()
        tr = correct / len(train)
        with torch.no_grad():
            v = sum(1 for s in valid
                    if int(torch.argmax(model.story_values(s)).item())
                    == exp.LABELS[s["answer"]]) / len(valid)
        print("    mode=%-4s seed=%d epoch %2d  train(run)=%.3f valid=%.3f"
              % (mode, seed, epoch, tr, v), flush=True)
        if tr >= 1.0 and v >= 1.0:
            print("    -> mode=%s seed=%d converged at epoch %d"
                  % (mode, seed, epoch), flush=True)
            return epoch
    return None


def diag_two_actor_sanity(data):
    print("== [5] 2-actor sanity (batch 1, lr 0.005) ==", flush=True)
    w2 = [s for s in data["train"] if s["width"] == 2]
    w2v = [s for s in data["validA"] if s["width"] == 2]
    print("  %d train / %d valid 2-actor stories" % (len(w2), len(w2v)),
          flush=True)
    for mode in ("logp", "raw"):
        ep = train_quick(w2, w2v, mode, seed=0, epochs=8)
        print("  mode=%-4s: %s" % (mode,
              ("100%% at epoch %d" % ep) if ep else "NOT converged in 8 epochs"),
              flush=True)


def diag_handcheck_speed():
    print("== [1b] hand-check story, lr 0.005, per-mode steps to correct ==",
          flush=True)
    story = {
        "width": 2, "actors": ["Alice", "Bob"],
        "events": [["walks", "Alice", "north"], ["walks", "Bob", "south"]],
        "question": ["Alice", "Bob"], "answer": "no", "n_ancilla": 0,
    }
    for mode in ("raw", "logp"):
        torch.manual_seed(12345)
        model = exp.Exp40Model(12345)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        steps_to_margin = None
        for step in range(1, 301):
            opt.zero_grad()
            vals = model.story_values(story)
            lo = logits_of(vals, mode)
            F.cross_entropy(lo.unsqueeze(0), torch.tensor([1])).backward()
            opt.step()
            with torch.no_grad():
                v = model.story_values(story)
            if v[1].item() > 2 * v[0].item() and steps_to_margin is None:
                steps_to_margin = step
                break
        print("  mode=%-4s: steps until p_no > 2*p_yes: %s"
              % (mode, steps_to_margin or ">300"), flush=True)


def main():
    data = taskgen.generate_dataset(40)
    diag_gradient_scale(data)
    diag_handcheck_speed()
    diag_two_actor_sanity(data)


if __name__ == "__main__":
    main()
