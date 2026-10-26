import json, numpy as np
from exp13_arabert_comparison import build_circuits, warmstart_weights
from lambeq import NumpyModel

data = json.load(open("sentences.json"))["WordOrderMatched"]
sents = [d["sentence"] for d in data[:2]]
circuits, valid = build_circuits(sents, 1, 1)
model = NumpyModel.from_diagrams(circuits, use_jit=False)
model.weights = warmstart_weights(model)
probs = np.asarray(model(circuits))
w = dict(zip([str(s) for s in model.symbols], model.weights))
for i, c in enumerate(circuits):
    syms = sorted(c.free_symbols, key=str)
    vals = [w[str(s)] for s in syms]
    res = c.lambdify(*syms)(*vals).eval()
    arr = np.asarray(res)
    print(f"circuit {i}: type={type(res).__name__} dtype={arr.dtype} shape={arr.shape}")
    print("  raw:", arr.flatten())
    p = np.sum(np.abs(arr)**2)
    print("  postselect norm p =", p)
    print("  |amp|^2/p =", (np.abs(arr.flatten())**2)/p, " vs model probs:", probs[i])
