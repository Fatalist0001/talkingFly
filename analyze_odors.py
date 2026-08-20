"""Inspect stage-3 results: which glomeruli drive each odor, spike rates.

Usage:
    python analyze_odors.py [--dir results_odor]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results_odor")
    args = p.parse_args()

    meta = pd.read_csv(os.path.join(args.dir, "neurons.csv"))
    states = np.load(os.path.join(args.dir, "states.npy"))  # (trials, neurons)
    with open(os.path.join(args.dir, "summary.json")) as f:
        summ = json.load(f)

    trials = summ["trials"]
    orn_mask = meta["is_orn"].to_numpy(dtype=bool)
    gl = meta["cell_type"].to_numpy()

    print(f"neurons: {len(meta)}  (ORNs: {orn_mask.sum()})   "
          f"trials: {len(states)}")
    print()
    for t, s in zip(trials, states):
        name = t["odor"]
        if name == "__blank__":
            print(f"--- {name}")
            continue
        print(f"--- {name} ({t['spikes_in_window']} spikes, "
              f"{t['active_in_window']} active)")
        # top glomeruli among ORNs
        sorn = np.where(orn_mask, s, 0)
        gsum = pd.Series(sorn, index=gl).groupby(level=0).sum().sort_values(
            ascending=False)
        top = gsum.head(6)
        print("   top glomeruli:", ", ".join(
            f"{g}:{int(v)}" for g, v in top.items() if g))
        # most active non-ORN neurons
        snon = np.where(~orn_mask, s, 0)
        idx = np.argsort(-snon)[:5]
        nid = meta["root_id"][k] if False else [int(r) for r in meta["root_id"]]
        print("   top non-ORN:", ", ".join(
            f"[{nid[k]}]{(meta['cell_type'][k] if isinstance(meta['cell_type'][k], str) and meta['cell_type'][k] != 'nan' else 'AL-neuron')}:{int(s[k])}"
            for k in idx if s[k] > 0))

    # time-resolved: population firing rate per 5ms bin per odor
    print()
    print("pop rate (Hz, all neurons) per 5 ms bin:")
    for t, (fn) in enumerate(sorted(os.listdir(args.dir),
                                    key=lambda x: int(x.split("_")[1][:-2]))
                            if False else [f"spikes_{k:02d}_t.npy"
                                           for k in range(len(trials))]):
        pass
    for k in range(len(trials)):
        ts = np.load(os.path.join(args.dir, f"spikes_{k:02d}_t.npy"))
        name = trials[k]["odor"]
        if ts.size == 0:
            print(f"  {name:22s} (no spikes)")
            continue
        bins = np.arange(0, ts.max() + 5, 5)
        hist, _ = np.histogram(ts, bins=bins)
        rate = hist / (5 / 1000) / len(meta)
        top = bins[np.argmax(hist)]
        print(f"  {name:22s} peak bin {top:6.1f} ms  "
              f"peak rate {rate.max():6.1f} Hz")


if __name__ == "__main__":
    main()