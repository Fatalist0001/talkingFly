"""Demo: are different odors distinguishable as ORN drive currents?"""
import numpy as np
import prepare_olfaction as op

orn, matrix, names = op.load()
print(f"{len(orn)} ORNs, {matrix.shape[1]} odors")
print("sample odors:", ", ".join(names[i] for i in [0, 2, 5, 10, 40]))
print()

demo = ["ethyl acetate", "isoamyl acetate", "citronellal", "benzaldehyde",
        "1-octanol", "ethyl butyrate"]
found = {name: op.find_odor(name, names) for name in demo}
for name, idx in found.items():
    print(f"{name:20s} -> idx={idx}  name='{names[idx] if idx >= 0 else '??'}'")
print()

valid = {k: v for k, v in found.items() if v >= 0}
idxs = list(valid.values())
clean = np.nan_to_num(matrix[:, idxs], nan=0.0)
pair = np.corrcoef(clean, rowvar=False)
labels = list(valid.keys())
print("pairwise correlation of ORN drive patterns:")
print(f"{'':20s}" + "".join(f"{l:>18s}"[:18] for l in labels))
for r, lab in zip(pair, labels):
    row = "".join(f"{x:>18.2f}"[:18] for x in r)
    print(f"{lab:20s}" + row)

print()
for name, idx in valid.items():
    vec = np.nan_to_num(matrix[:, idx], nan=0.0)
    print(f"{name:24s} driven ORNs (resp>0.5): {(vec>0.5).sum():4d}, "
          f" total drive: {vec.sum():6.1f}")