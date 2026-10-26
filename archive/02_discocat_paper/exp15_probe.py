import json, numpy as np
from exp13_arabert_comparison import build_circuits, warmstart_weights
from lambeq import NumpyModel

data = json.load(open("sentences.json"))["WordOrderMatched"]
sents = [d["sentence"] for d in data[:4]]
print("sentences:", sents)

for L in (0, 1):
    circuits, valid = build_circuits(sents, L, 1)
    print(f"L{L}: {len(circuits)} circuits built, valid_idx={valid}")
    model = NumpyModel.from_diagrams(circuits, use_jit=False)
    model.weights = warmstart_weights(model)
    out = np.asarray(model(circuits))
    print(f"L{L} model() -> dtype={out.dtype} shape={out.shape}")
    print(out)
    try:
        c = circuits[0]
        f = c.lambdify(*model.symbols)
        res = f(*model.weights).eval()
        arr = np.asarray(res.array)
        print(f"L{L} lambdify amp: dtype={arr.dtype} shape={arr.shape}")
        print(arr)
    except Exception as e:
        print(f"L{L} lambdify route failed: {type(e).__name__}: {e}")
