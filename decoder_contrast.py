"""Stage 4.7: contrastive (metric) learning for odor-state embeddings.

Lever 1 (contrastive objective) + lever 2 (temporal trajectory features) on top
of the stronger drive regime (lever 3, gain=40).

Instead of "image -> odor name" (closed-set classifier that cannot generalise to
unseen odors), we learn an encoder f: brain_state -> z such that
    same odor   -> close in z
    diff odor    -> far in z
via an InfoNCE (contrastive) loss over training-odor trials.

Generalisation test: hold out 9 odors; encode their trials; find the nearest
*seen* odor centroid in the learned space; check whether that seen odor is among
the chemically (DoOR) most-similar to the held-out odor. Report hit@k vs chance
and mean DoOR-rank of the retrieved neighbor. Compare against a non-contrastive
PCA baseline (same protocol, no learned metric).

Run:  python decoder_contrast.py
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

import prepare_olfaction as op


def pca_feats(feats, train_mask, k=128):
    pca = PCA(n_components=k, whiten=True)
    Ztr = pca.fit_transform(feats[train_mask])
    Z = pca.transform(feats)
    return Z, pca


def sample_positives(y, train_idx):
    """For each training trial, pick another trial of the SAME odor as positive."""
    ytr = y[train_idx]
    pos = np.empty(len(train_idx), dtype=int)
    for l in np.unique(ytr):
        ids = np.where(ytr == l)[0]
        for gi in ids:
            choices = ids[ids != gi]
            pos[gi] = np.random.choice(choices)
    return pos


def info_nce_loss_dW(Z, anchor_pos, W, tau=0.1):
    """InfoNCE over a batch. anchor_pos[k] = index of positive for anchor k.

    Returns (loss, dW). Embeddings are L2-normalised; W: (K -> E).
    """
    m = Z.shape[0]
    h = Z @ W
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-9)
    p = h / nrm                                  # (m, E) normalized
    sim = p @ p.T / tau                          # (m, m)
    # softmax over rows
    sim = sim - sim.max(axis=1, keepdims=True)
    e = np.exp(sim)
    row_sum = e.sum(axis=1, keepdims=True)
    sm = e / row_sum
    labels = np.arange(m)
    loss = float(-np.log(sm[labels, labels] + 1e-12).mean())
    g = sm.copy()
    g[labels, labels] -= 1.0                     # (softmax - onehot)
    g = g / tau                                  # d loss / d sim
    S = (g + g.T) @ p                            # row k = sum_j (G[k,j]+G[j,k]) p_j
    ps = np.sum(p * S, axis=1, keepdims=True)    # p_k . S_k
    grad_h = (S - p * ps) / nrm                  # (m, E)
    dW = Z.T @ grad_h                            # (K, E)
    return loss, dW


def train_contrastive(Z, y, train_idx, E=64, tau=0.1, lr=0.3, epochs=300,
                      seed=0):
    rng = np.random.default_rng(seed)
    K = Z.shape[1]
    W = (rng.standard_normal((K, E)) * 0.01).astype(np.float32)
    order = np.arange(len(train_idx))
    for ep in range(epochs):
        rng.shuffle(order)
        pos_local = sample_positives(y, train_idx[order])
        loss, dW = info_nce_loss_dW(Z[train_idx[order]], train_idx[order][pos_local], W, tau)
        W = W - lr * dW
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  InfoNCE loss = {loss:.4f}")
    return W


def embed(Z, W):
    h = Z @ W
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    return h / np.maximum(nrm, 1e-9)


def train_relational(Z, y, train_idx, odor_idx, matrix, E=64, lr=0.3,
                     epochs=400, seed=0):
    """Metric-matching: learn embeddings whose inner product == DoOR similarity.

    This bakes the chemical (DoOR) structure into the brain-state space, so a
    held-out odor's embedding lands near its DoOR-similar seen odor.
    """
    rng = np.random.default_rng(seed)
    K = Z.shape[1]
    W = (rng.standard_normal((K, E)) * 0.01).astype(np.float32)
    # target DoOR correlation among training (seen) odors
    seen = np.unique(y[train_idx])
    corr = np.zeros((len(seen), len(seen)))
    for a, la in enumerate(seen):
        for b, lb in enumerate(seen):
            ca = np.nan_to_num(matrix[:, odor_idx[la]])
            cb = np.nan_to_num(matrix[:, odor_idx[lb]])
            if ca.std() == 0 or cb.std() == 0:
                corr[a, b] = 0.0
            else:
                corr[a, b] = np.corrcoef(ca, cb)[0, 1]
    pos = {l: i for i, l in enumerate(seen)}
    m = len(train_idx)
    T = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            T[i, j] = corr[pos[y[train_idx[i]]], pos[y[train_idx[j]]]]
    for ep in range(epochs):
        h = Z[train_idx] @ W
        nrm = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        z = h / nrm
        sim = z @ z.T
        diff = sim - T
        loss = float(np.mean(diff ** 2))
        G = 2.0 * (diff + diff.T)                 # d loss / d sim (symmetrized)
        S = G @ z                                # row k = sum_j G[k,j] z_j
        ps = np.sum(z * S, axis=1, keepdims=True)
        grad_h = (S - z * ps) / nrm
        W = W - lr * (Z[train_idx].T @ grad_h)
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  relational MSE = {loss:.4f}")
    return W



def doorgt(held_labels, seen_labels, matrix, names, odor_idx):
    """For each held-out odor, rank of each seen odor by DoOR similarity."""
    rank = {}
    for hl in held_labels:
        hv = matrix[:, odor_idx[hl]]
        corr = np.array([np.corrcoef(hv, matrix[:, odor_idx[sl]])[0, 1]
                         for sl in seen_labels])
        order = np.argsort(-corr)              # positions into seen_labels
        rank[hl] = {seen_labels[pos]: int(r) for r, pos in enumerate(order)}
    return rank


def eval_generalisation(emb, y, held_labels, seen_labels, gt_rank, k_list):
    held_trial_idx = np.concatenate([np.where(y == hl)[0] for hl in held_labels])
    # centroids of seen odors
    cents = {l: emb[y == l].mean(0) for l in seen_labels}
    ranks = []
    for ht in held_trial_idx:
        q = emb[ht]
        d = {l: float(q @ c) for l, c in cents.items()}
        ri = max(d, key=d.get)
        ranks.append(gt_rank[y[ht]][ri])
    ranks = np.array(ranks)
    out = {k: float(np.mean(ranks < k)) for k in k_list}
    out["mean_rank"] = float(np.mean(ranks))
    out["chance@1"] = 1.0 / len(seen_labels)
    return out


def run_feature(name, feats, y, held_labels, seen_labels, gt_rank, args,
                matrix, odor_idx):
    print(f"\n=== feature: {name} ===")
    train_mask = np.isin(y, seen_labels)
    Z, _ = pca_feats(feats, train_mask, k=args.pca)
    # raw PCA baseline
    raw_emb = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    raw_res = eval_generalisation(raw_emb, y, held_labels, seen_labels,
                                  gt_rank, args.k)
    print(f"  RAW PCA     held-out: " +
          "  ".join(f"hit@{k}={raw_res[k]:.2f}" for k in args.k) +
          f"  mean_rank={raw_res['mean_rank']:.1f} "
          f"(chance@1={raw_res['chance@1']:.3f})")
    # contrastive (plain: all-different-odors equally negative)
    train_idx = np.where(train_mask)[0]
    W = train_contrastive(Z, y, train_idx, E=args.emb, tau=args.tau,
                          lr=args.lr, epochs=args.epochs, seed=args.seed)
    c_emb = embed(Z, W)
    c_res = eval_generalisation(c_emb, y, held_labels, seen_labels,
                                gt_rank, args.k)
    print(f"  CONTRASTIVE held-out: " +
          "  ".join(f"hit@{k}={c_res[k]:.2f}" for k in args.k) +
          f"  mean_rank={c_res['mean_rank']:.1f} "
          f"(chance@1={c_res['chance@1']:.3f})")
    # relational (similarity-aware: inner product == DoOR correlation)
    Wr = train_relational(Z, y, train_idx, odor_idx, matrix, E=args.emb,
                          lr=args.lr, epochs=args.epochs, seed=args.seed)
    r_emb = embed(Z, Wr)
    r_res = eval_generalisation(r_emb, y, held_labels, seen_labels,
                                gt_rank, args.k)
    print(f"  RELATIONAL   held-out: " +
          "  ".join(f"hit@{k}={r_res[k]:.2f}" for k in args.k) +
          f"  mean_rank={r_res['mean_rank']:.1f} "
          f"(chance@1={r_res['chance@1']:.3f})")
    # closed-set sanity (seen odors, 1-NN in embedding)
    seen_trial = np.concatenate([np.where(y == l)[0] for l in seen_labels])
    for tag, emb in (("contrastive", c_emb), ("relational", r_emb)):
        cents = {l: emb[y == l].mean(0) for l in seen_labels}
        closed = [max(cents, key=lambda l: emb[st] @ cents[l])
                  for st in seen_trial]
        acc = float(np.mean([closed[i] == y[seen_trial][i]
                             for i in range(len(seen_trial))]))
        print(f"  closed-set 1-NN acc ({tag}, {len(seen_labels)} odors): {acc:.3f}")
    return {"raw": raw_res, "contrastive": c_res, "relational": r_res}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="decoder/dataset.npz")
    p.add_argument("--pca", type=int, default=128)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--out", default="decoder/contrast.json")
    args = p.parse_args()

    d = np.load(args.data)
    y = d["y"]
    X_counts = d["X_counts"].astype(float)
    X_traj = d["X_traj"].astype(float).reshape(len(y), -1)
    names = json.load(open("decoder/odor_names.json"))
    n_odor = len(names) - 1

    _, matrix, door_names = op.load()
    odor_idx = [door_names.index(n) for n in names[:n_odor]]

    # held-out split: every 4th odor (9 held, 27 seen)
    held_labels = list(range(0, n_odor, 4))
    seen_labels = [l for l in range(n_odor) if l not in held_labels]
    gt_rank = doorgt(held_labels, seen_labels, matrix, names, odor_idx)
    print(f"held-out odors ({len(held_labels)}): "
          f"{[names[i] for i in held_labels]}")

    results = {}
    results["counts"] = run_feature("X_counts", X_counts, y, held_labels,
                                    seen_labels, gt_rank, args, matrix,
                                    odor_idx)
    results["traj"] = run_feature("X_traj", X_traj, y, held_labels,
                                  seen_labels, gt_rank, args, matrix, odor_idx)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
