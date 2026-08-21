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
    oi, corr = _odor_corr(y, train_idx, odor_idx, matrix)
    T = corr[np.ix_(oi, oi)]
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


def _odor_corr(y, train_idx, odor_idx, matrix):
    """Pearson corr of DoOR vectors among the seen odors, expanded to trial pairs.

    Returns (oi, corr_SxS): oi = odor-position per training trial, corr_SxS the
    (n_seen x n_seen) correlation matrix. Trial-pair matrix is corr_SxS[ix_(oi,oi)].
    Vectorised over odors (not trials) so it scales to thousands of trials.
    """
    seen = np.unique(y[train_idx])
    pos = {l: i for i, l in enumerate(seen)}
    D = np.nan_to_num(matrix[:, [odor_idx[l] for l in seen]])   # (78, n_seen)
    Dc = D - D.mean(axis=0, keepdims=True)
    Dn = Dc / np.maximum(np.linalg.norm(Dc, axis=0, keepdims=True), 1e-9)
    corr = Dn.T @ Dn
    oi = np.array([pos[o] for o in y[train_idx]])
    return oi, corr


def train_rank(Z, y, train_idx, odor_idx, matrix, E=64, lr=0.3, epochs=300,
               seed=0, temp_s=0.2, tau=0.1):
    """Soft (supervised) contrastive: preserve DoOR *similarity ranking*.

    Anchor trial's target over other trials is a softmax over DoOR correlations,
    not a hard "same odor vs different". Pulls chemically-similar odors together
    more strongly than dissimilar ones (a sharper metric than the relational MSE).
    """
    rng = np.random.default_rng(seed)
    K = Z.shape[1]
    W = (rng.standard_normal((K, E)) * 0.01).astype(np.float32)
    oi, corr = _odor_corr(y, train_idx, odor_idx, matrix)
    C = corr[np.ix_(oi, oi)]
    Cs = C / temp_s
    Cs = Cs - Cs.max(axis=1, keepdims=True)
    e = np.exp(Cs)
    S = e / e.sum(axis=1, keepdims=True)          # soft targets
    for ep in range(epochs):
        h = Z[train_idx] @ W
        nrm = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        z = h / nrm
        sim = z @ z.T
        ls = sim / tau
        ls = ls - ls.max(axis=1, keepdims=True)
        ep_ = np.exp(ls)
        P = ep_ / ep_.sum(axis=1, keepdims=True)
        loss = float(np.mean(-np.sum(S * np.log(P + 1e-12), axis=1)))
        G = (P - S) / tau
        Sv = (G + G.T) @ z
        ps = np.sum(z * Sv, axis=1, keepdims=True)
        grad_h = (Sv - z * ps) / nrm
        W = W - lr * (Z[train_idx].T @ grad_h)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  rank KL = {loss:.4f}")
    return W


def train_reg78(Z, y, train_idx, odor_idx, matrix, E=78, lr=0.1, epochs=400,
                seed=0):
    """Linear map brain PCA -> full 78-receptor DoOR vector (per trial's odor).

    Embedding = predicted DoOR vector; retrieval then directly tests whether the
    brain state recovers enough chemical identity to land near its true neighbor.
    """
    rng = np.random.default_rng(seed)
    K = Z.shape[1]
    W = (rng.standard_normal((K, E)) * 0.01).astype(np.float32)
    m = len(train_idx)
    T = np.zeros((m, matrix.shape[0]))
    for i in range(m):
        T[i] = np.nan_to_num(matrix[:, odor_idx[y[train_idx[i]]]])
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    for ep in range(epochs):
        h = Z[train_idx] @ W
        nrm = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        pred = h / nrm
        loss = float(np.mean((pred - T) ** 2))
        grad_pred = 2.0 * (pred - T)
        grad_h = (grad_pred - pred * np.sum(pred * grad_pred, axis=1, keepdims=True)) / nrm
        W = W - lr * (Z[train_idx].T @ grad_h)
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  reg78 MSE = {loss:.4f}")
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


def expand_features(feats, n, seed):
    """Fixed random expansion (ReLU) — a cheap proxy for MB expansion recoding.

    Maps the compressed AL state into a high-dimensional sparse space; if this
    lifts held-out retrieval above chance, a real mushroom-body layer is warranted.
    """
    rng = np.random.default_rng(seed)
    D = feats.shape[1]
    R = rng.standard_normal((D, n)).astype(np.float32)
    return np.maximum(feats @ R, 0.0).astype(np.float32)


def run_feature(name, feats, y, held_labels, seen_labels, gt_rank, args,
                matrix, odor_idx):
    print(f"\n=== feature: {name} ===")
    if args.expand and args.expand > 0:
        feats = expand_features(feats, args.expand, args.seed)
    train_mask = np.isin(y, seen_labels)
    Z, _ = pca_feats(feats, train_mask, k=args.pca)
    train_idx = np.where(train_mask)[0]

    embs = {}
    embs["RAW_PCA"] = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    Wc = train_contrastive(Z, y, train_idx, E=args.emb, tau=args.tau,
                           lr=args.lr, epochs=args.epochs, seed=args.seed)
    embs["CONTRASTIVE"] = embed(Z, Wc)
    Wr = train_relational(Z, y, train_idx, odor_idx, matrix, E=args.emb,
                          lr=args.lr, epochs=args.epochs, seed=args.seed)
    embs["RELATIONAL"] = embed(Z, Wr)
    Wk = train_rank(Z, y, train_idx, odor_idx, matrix, E=args.emb,
                    lr=args.lr, epochs=args.epochs, seed=args.seed,
                    temp_s=args.temp_s, tau=args.tau)
    embs["RANK"] = embed(Z, Wk)
    Wg = train_reg78(Z, y, train_idx, odor_idx, matrix, E=matrix.shape[0],
                     lr=args.lr, epochs=args.epochs, seed=args.seed)
    embs["REG78"] = embed(Z, Wg)

    out = {}
    for tag, emb in embs.items():
        res = eval_generalisation(emb, y, held_labels, seen_labels, gt_rank,
                                  args.k)
        out[tag] = res
        print(f"  {tag:12s} held-out: " +
              "  ".join(f"hit@{kk}={res[kk]:.2f}" for kk in args.k) +
              f"  mean_rank={res['mean_rank']:.1f} "
              f"(chance@1={res['chance@1']:.3f})")
    seen_trial = np.concatenate([np.where(y == l)[0] for l in seen_labels])
    for tag, emb in embs.items():
        cents = {l: emb[y == l].mean(0) for l in seen_labels}
        closed = [max(cents, key=lambda l: emb[st] @ cents[l])
                  for st in seen_trial]
        acc = float(np.mean([closed[i] == y[seen_trial][i]
                             for i in range(len(seen_trial))]))
        print(f"  {tag:12s} closed-set 1-NN acc ({len(seen_labels)}): {acc:.3f}")
    return out


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
    p.add_argument("--held-every", type=int, default=4,
                   help="hold out every Nth odor (N=4 -> 1/4 unseen)")
    p.add_argument("--temp-s", type=float, default=0.2,
                   help="soft-target temperature for the ranking loss")
    p.add_argument("--expand", type=int, default=0,
                   help="random ReLU expansion dim (MB expansion-recoding proxy)")
    p.add_argument("--orn-only", action="store_true",
                   help="decode from ORN spikes only (input layer, most identity)")
    p.add_argument("--out", default="decoder/contrast.json")
    args = p.parse_args()

    d = np.load(args.data)
    y = d["y"]
    X_counts = d["X_counts"].astype(float)
    is_orn = d["is_orn"].astype(bool)
    if args.orn_only:
        X_counts = X_counts[:, is_orn]
    X_traj = (d["X_traj"].astype(float).reshape(len(y), -1)
              if "X_traj" in d.files else None)
    data_dir = os.path.dirname(args.data) or "."
    names = json.load(open(os.path.join(data_dir, "odor_names.json")))
    n_odor = len(names) - 1

    _, matrix, door_names = op.load()
    odor_idx = [door_names.index(n) for n in names[:n_odor]]

    # held-out split: every Nth odor
    held_labels = list(range(0, n_odor, args.held_every))
    seen_labels = [l for l in range(n_odor) if l not in held_labels]
    gt_rank = doorgt(held_labels, seen_labels, matrix, names, odor_idx)
    print(f"held-out odors ({len(held_labels)}): "
          f"{[names[i] for i in held_labels]}")

    results = {}
    results["counts"] = run_feature("X_counts", X_counts, y, held_labels,
                                    seen_labels, gt_rank, args, matrix, odor_idx)
    if X_traj is not None:
        results["traj"] = run_feature("X_traj", X_traj, y, held_labels,
                                      seen_labels, gt_rank, args, matrix,
                                      odor_idx)
    else:
        print("\n=== feature: X_traj ===\n  (not present in dataset, skipped)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
