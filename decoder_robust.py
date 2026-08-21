"""Stage 4.3 (robustness): how well do the odor decoders survive perturbations?

Closes the open "[ ] Устойчивость (шум / пропажа нейронов / укорочение окна)".

We take the best pipeline from 4.7 (relational contrastive on X_counts, and on
X_traj for the window test) and re-run it under three perturbations applied to the
*features* (so PCA/encoder are re-fit on the perturbed data — a true robustness
test, not a train/clean eval/perturbed mismatch):

  - noise      : Gaussian noise, std = frac * per-feature std
  - drop       : randomly zero a fraction of NEURONS (columns / all time bins)
  - window     : (X_traj only) keep only the first K trajectory bins

For each severity we report closed-set 1-NN acc (seen odors) and held-out
retrieval mean_rank (lower = better; random ~13 of 27). Degradation vs the
unperturbed baseline tells us how fragile the decoder is.

Run:  python decoder_robust.py
"""
import argparse
import json

import numpy as np

import prepare_olfaction as op
from decoder_contrast import (pca_feats, train_relational, embed,
                              eval_generalisation, doorgt)


def pipeline(feats, y, held_labels, seen_labels, gt_rank, args, matrix,
             odor_idx):
    train_mask = np.isin(y, seen_labels)
    Z, _ = pca_feats(feats, train_mask, k=args.pca)
    train_idx = np.where(train_mask)[0]
    Wr = train_relational(Z, y, train_idx, odor_idx, matrix, E=args.emb,
                          lr=args.lr, epochs=args.epochs, seed=args.seed)
    r_emb = embed(Z, Wr)
    res = eval_generalisation(r_emb, y, held_labels, seen_labels, gt_rank,
                              args.k)
    seen_trial = np.concatenate([np.where(y == l)[0] for l in seen_labels])
    cents = {l: r_emb[y == l].mean(0) for l in seen_labels}
    closed = [max(cents, key=lambda l: r_emb[st] @ cents[l])
              for st in seen_trial]
    acc = float(np.mean([closed[i] == y[seen_trial][i]
                         for i in range(len(seen_trial))]))
    return res, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="decoder/dataset.npz")
    p.add_argument("--pca", type=int, default=128)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--out", default="decoder/robust.json")
    args = p.parse_args()

    d = np.load(args.data)
    y = d["y"]
    X_counts = d["X_counts"].astype(float)
    X_traj = d["X_traj"].astype(float)
    meta = json.load(open("decoder/meta.json"))
    TB = meta["traj_bins"]
    N = meta["n_neurons"]
    names = json.load(open("decoder/odor_names.json"))
    n_odor = len(names) - 1

    _, matrix, door_names = op.load()
    odor_idx = [door_names.index(n) for n in names[:n_odor]]

    held_labels = list(range(0, n_odor, 4))
    seen_labels = [l for l in range(n_odor) if l not in held_labels]
    gt_rank = doorgt(held_labels, seen_labels, matrix, names, odor_idx)
    rng = np.random.default_rng(args.seed)

    results = {}

    # ---- X_counts: noise + neuron dropout ------------------------------------
    print("=== X_counts: noise ===")
    noise_res = {}
    for frac in [0.0, 0.1, 0.25, 0.5, 1.0]:
        f = X_counts.copy()
        if frac > 0:
            f = f + frac * f.std(0, keepdims=True) * rng.standard_normal(f.shape)
        res, acc = pipeline(f, y, held_labels, seen_labels, gt_rank, args,
                            matrix, odor_idx)
        noise_res[str(frac)] = {"mean_rank": res["mean_rank"],
                                "hit@1": res[1], "closed_acc": acc}
        print(f"  noise={frac:4.2f}  held-out mean_rank={res['mean_rank']:.1f} "
              f"hit@1={res[1]:.2f}  closed_acc={acc:.3f}")
    results["counts_noise"] = noise_res

    print("=== X_counts: neuron dropout ===")
    drop_res = {}
    for frac in [0.0, 0.1, 0.3, 0.5, 0.7]:
        f = X_counts.copy()
        if frac > 0:
            keep = rng.random(N) > frac
            f[:, ~keep] = 0.0
        res, acc = pipeline(f, y, held_labels, seen_labels, gt_rank, args,
                            matrix, odor_idx)
        drop_res[str(frac)] = {"mean_rank": res["mean_rank"],
                               "hit@1": res[1], "closed_acc": acc}
        print(f"  drop={frac:4.2f}  held-out mean_rank={res['mean_rank']:.1f} "
              f"hit@1={res[1]:.2f}  closed_acc={acc:.3f}")
    results["counts_drop"] = drop_res

    # ---- X_traj: noise + neuron dropout + shortened window -------------------
    print("=== X_traj: noise ===")
    traj_noise = {}
    for frac in [0.0, 0.1, 0.25, 0.5]:
        f = X_traj.reshape(len(y), TB, N).copy()
        if frac > 0:
            f = f + frac * f.std(0, keepdims=True) * rng.standard_normal(f.shape)
        res, acc = pipeline(f.reshape(len(y), -1), y, held_labels, seen_labels,
                            gt_rank, args, matrix, odor_idx)
        traj_noise[str(frac)] = {"mean_rank": res["mean_rank"],
                                 "hit@1": res[1], "closed_acc": acc}
        print(f"  noise={frac:4.2f}  held-out mean_rank={res['mean_rank']:.1f} "
              f"hit@1={res[1]:.2f}  closed_acc={acc:.3f}")
    results["traj_noise"] = traj_noise

    print("=== X_traj: neuron dropout ===")
    traj_drop = {}
    for frac in [0.0, 0.1, 0.3, 0.5]:
        f = X_traj.reshape(len(y), TB, N).copy()
        if frac > 0:
            keep = rng.random(N) > frac
            f[:, :, ~keep] = 0.0
        res, acc = pipeline(f.reshape(len(y), -1), y, held_labels, seen_labels,
                            gt_rank, args, matrix, odor_idx)
        traj_drop[str(frac)] = {"mean_rank": res["mean_rank"],
                                "hit@1": res[1], "closed_acc": acc}
        print(f"  drop={frac:4.2f}  held-out mean_rank={res['mean_rank']:.1f} "
              f"hit@1={res[1]:.2f}  closed_acc={acc:.3f}")
    results["traj_drop"] = traj_drop

    print("=== X_traj: shortened window (first K bins) ===")
    traj_win = {}
    for K in [TB, 20, 10, 5]:
        f = X_traj.reshape(len(y), TB, N)[:, :K, :].reshape(len(y), -1)
        res, acc = pipeline(f, y, held_labels, seen_labels, gt_rank, args,
                            matrix, odor_idx)
        traj_win[str(K)] = {"mean_rank": res["mean_rank"],
                            "hit@1": res[1], "closed_acc": acc}
        print(f"  K={K:2d}  held-out mean_rank={res['mean_rank']:.1f} "
              f"hit@1={res[1]:.2f}  closed_acc={acc:.3f}")
    results["traj_window"] = traj_win

    with open(args.out, "w") as fo:
        json.dump(results, fo, indent=2)
    print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
