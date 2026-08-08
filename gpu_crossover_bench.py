"""Where, if anywhere, does a GPU beat the CPU for THIS workload?

The exp40-43 inner loop is a chain of small dense matmuls: per event, a 4x4 verb
block applied to a 2^k statevector (k = 4-6 referents, so 16-64 dims), ~6 events
per story, then a short adjoint chain for the question. At batch=1 that is a long
SEQUENTIAL dependency chain of tiny ops -- the regime where accelerators lose,
because a kernel launch costs more than the arithmetic it dispatches.

The only way a GPU can win is if the parallelism axis changes: instead of one
model on one story, run R independent replicas (restarts / seeds / configs / arms)
as a leading batch dimension, so each launch does R times the work. exp43a ran 40
restarts; Phase 2 runs 5 seeds x 6 arms = 30 -- both are embarrassingly parallel
and identically shaped, so this is a real option, not a hypothetical.

This measures the crossover R at each state dimension the project uses or plans
to use. Nothing here touches the research code or its results.

Replica counts are capped per dimension to keep block tensors under ~1.5 GB.
"""
import time

import torch

torch.set_num_threads(1)

N_EVENTS = 6
N_STEPS = 25
REPS = 8
MEM_BUDGET = 1.5e9


def bench(device, R, dim):
    dev = torch.device(device)
    psi = torch.randn(R, dim, device=dev)
    blocks = torch.randn(R, N_EVENTS, dim, dim, device=dev) * 0.3
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    for _ in range(2):
        x = psi
        for e in range(N_EVENTS):
            x = torch.bmm(blocks[:, e], x.unsqueeze(-1)).squeeze(-1)
    sync()
    t0 = time.perf_counter()
    for _ in range(REPS):
        x = psi
        for _ in range(N_STEPS):
            for e in range(N_EVENTS):
                x = torch.bmm(blocks[:, e], x.unsqueeze(-1)).squeeze(-1)
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    sync()
    del psi, blocks
    if device == "mps":
        torch.mps.empty_cache()
    return (time.perf_counter() - t0) / REPS * 1000.0


def main():
    have_mps = torch.backends.mps.is_available()
    print("torch %s   mps available: %s" % (torch.__version__, have_mps),
          flush=True)
    print("chain = %d events x %d steps, timed over %d reps\n"
          % (N_EVENTS, N_STEPS, REPS), flush=True)
    for dim, label in ((16, "dim=16   K=4 referents -- THE CURRENT REGIME (exp42/43)"),
                       (64, "dim=64   K=6 referents -- L2 rung scale"),
                       (1024, "dim=1024 K=10 referents -- exp34b scaling territory")):
        rmax = MEM_BUDGET / (N_EVENTS * dim * dim * 4)
        rs = [r for r in (1, 8, 30, 200, 1000, 5000, 20000) if r <= rmax]
        print(label, flush=True)
        print("  %9s %11s %11s %11s" % ("replicas", "cpu ms", "mps ms", "cpu/mps"),
              flush=True)
        for R in rs:
            c = bench("cpu", R, dim)
            m = bench("mps", R, dim) if have_mps else float("nan")
            flag = "   <-- GPU wins" if have_mps and m < c else ""
            print("  %9d %11.2f %11.2f %10.2fx%s" % (R, c, m, c / m, flag),
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
