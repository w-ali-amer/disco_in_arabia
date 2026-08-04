import io
src = io.open("exp28a_real_gates.py", encoding="utf-8").read()
old = """        else:  # 2-qubit
            for wid in (wires[off], wires[off + 1]):
                info[wid]["ops"].append(oi)
    return info, wires  # wires = ids still open at end (the s wire(s))"""
new = """        else:  # 2-qubit
            for wid in (wires[off], wires[off + 1]):
                info[wid]["ops"].append(oi)
            if kind == "SWAP":
                # VSO circuits contain Swap: the tracker must exchange wire
                # IDENTITIES or every downstream wire is mislabeled
                # (caught by the closure gate)
                wires[off], wires[off + 1] = wires[off + 1], wires[off]
    return info, wires  # wires = ids still open at end (the s wire(s))"""
assert src.count(old) == 1
io.open("exp28a_real_gates.py", "w", encoding="utf-8").write(src.replace(old, new))
print("PATCHED")
