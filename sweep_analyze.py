"""Analyze odor-state separability across drive-sweep datasets.

For each dataset under a sweep dir, report:
  - min_odor_corr : min pairwise correlation of odor MEAN states
                    (LOWER = odors more distinct; need this well below 0.92)
  - mean_intra    : mean within-odor trial correlation (reproducibility)
  - blank_corr    : mean |corr| between blank mean state and odor means
                    (LOWER = blank cleanly separable from odors)
"""
import glob
import json
import os

import numpy as np


def analyze(d):
    data = np.load(os.path.join(d, "dataset.npz"))
    y = data["y"]
    names = json.load(open(os.path.join(d, "odor_names.json")))
    n_odor = len(names) - 1
    X = data["X_counts"].astype(float)
    mask = y < n_odor
    Xo, lo = X[mask], y[mask]
    means = np.stack([Xo[lo == l].mean(0) for l in range(n_odor)])

    c = np.corrcoef(means)
    off = c[np.triu_indices(n_odor, 1)]
    min_odor = float(np.abs(off).min())

    intra = []
    for l in range(n_odor):
        vs = Xo[lo == l]
        if len(vs) > 1:
            intra.append(np.corrcoef(vs)[np.triu_indices(len(vs), 1)].mean())
    mean_intra = float(np.nanmean(intra))

    blank_mean = X[y == n_odor].mean(0)
    bcorr = np.array([np.corrcoef(blank_mean, means[l])[0, 1]
                      for l in range(n_odor)])
    blank_corr = float(np.mean(np.abs(bcorr)))
    return min_odor, mean_intra, blank_corr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sweepdir")
    args = ap.parse_args()
    print(f"{'dataset':28s} {'min_odor_corr':>13s} {'mean_intra':>10s} "
          f"{'blank_corr':>11s}")
    for d in sorted(glob.glob(os.path.join(args.sweepdir, "*"))):
        if not os.path.isdir(d):
            continue
        mo, mi, bc = analyze(d)
        print(f"{os.path.basename(d):28s} {mo:13.3f} {mi:10.3f} {bc:11.3f}")


if __name__ == "__main__":
    main()