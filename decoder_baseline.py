"""Stage 4.2 - baseline decoders on decoder/dataset.npz.

Closed-set evaluation (train/test split BY TRIALS within the same odor set,
stratified on label) and generalization-to-held-out-odors evaluation (train on
a subset of odors, probe whether unseen-odor trials land near their expected
DoOR-most-similar seen odor).

Usage:
  .\\.venv\\Scripts\\python.exe decoder_baseline.py [--outdir decoder]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 0
N_HELD = 9          # odors held out in the generalization test
N_SPLITS = 3
TEST_FRAC = 0.25


def load(outdir):
    d = np.load(os.path.join(outdir, "dataset.npz"))
    names = json.load(open(os.path.join(outdir, "odor_names.json")))
    meta = json.load(open(os.path.join(outdir, "meta.json")))
    all_names = json.load(open("olfactory/odor_names.json"))
    resp = np.load("olfactory/resp_matrix.npy")          # (n_orn, 691)
    return d, names, meta, all_names, resp


def make_reps(d):
    n = d["X_bins"].shape[0]
    return {
        "counts": d["X_counts"].astype(np.float64),
        "bins": d["X_bins"].reshape(n, -1).astype(np.float64),
        "glom": d["X_glom"].astype(np.float64),
        "glom_bins": d["X_glom_bins"].reshape(n, -1).astype(np.float64),
    }


def clfs():
    return {
        "logreg": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs"),
        "linear_svm": SGDClassifier(
            loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-4,
            random_state=SEED),
    }


def corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def topk_hits(y_true, sorted_ranks, k):
    return (sorted_ranks[:, :k] == y_true[:, None]).any(axis=1).mean()


def closed_set(X, y, n_cls):
    """Stratified trial-level split inside the SAME odor set."""
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=SEED)
    out = {}
    for cname, clf in clfs().items():
        acc1, acc3, blank_acc = [], [], []
        for tr, te in sss.split(X, y):
            scl = StandardScaler().fit(X[tr])
            Xtr, Xte = scl.transform(X[tr]), scl.transform(X[te])
            m = clf.__class__(**clf.get_params())
            m.fit(Xtr, y[tr])
            if hasattr(m, "predict_proba"):
                scores = m.predict_proba(Xte)
            else:
                scores = m.decision_function(Xte)
            order = np.argsort(-scores, axis=1)
            acc1.append((order[:, 0] == y[te]).mean())
            acc3.append(topk_hits(y[te], order, 3))
            blank_te = y[te] == n_cls - 1
            if blank_te.any():
                blank_acc.append((order[blank_te, 0] == n_cls - 1).mean())
        out[cname] = {
            "acc@1": float(np.mean(acc1)),
            "acc@3": float(np.mean(acc3)),
            "blank_acc": float(np.mean(blank_acc)),
        }
    return out


def heldout_metrics(X, y, n_odor, names, all_names, resp, n_cls):
    """Train-free prototype probe for unseen odors.

    Prototypes are trial-mean z-scored states of seen odors.  For each trial
    of a held-out odor we take its nearest seen prototype; we then score how
    often that prototype is the seen odor that is most receptor-similar
    (DoOR vector cosine) to the presented odor.
    """
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(n_odor, N_HELD, replace=False))
    seen = np.array([c for c in range(n_odor) if c not in held])

    col = [all_names.index(names[c]) for c in range(n_odor)]
    R = np.nan_to_num(resp[:, col], nan=0.0).astype(np.float64)
    exp_neighbor = {}
    for c in held:
        inter = np.argsort([-corr2(R[:, c], R[:, s]) for s in seen])[0]
        exp_neighbor[int(c)] = int(seen[inter])

    scl = StandardScaler().fit(X)
    Z = scl.transform(X)
    proto = np.stack([Z[y == c].mean(0) for c in seen])   # (n_seen, n_feat)
    # pairwise Euclidean without materialising the (n x n_seen x f) tensor
    dist = np.empty((Z.shape[0], len(seen)))
    for i, p in enumerate(proto):
        d = Z - p
        dist[:, i] = np.sqrt(np.einsum("ij,ij->i", d, d))

    hit1, hit3, cnt = 0.0, 0.0, 0
    for c in held:
        rows = np.where(y == c)[0]
        if rows.size == 0:
            continue
        rk = np.argsort(dist[rows], axis=1)
        hit1 += (rk[:, 0] == exp_neighbor[int(c)]).sum()
        hit3 += (rk[:, :3] == exp_neighbor[int(c)]).any(axis=1).sum()
        cnt += rows.size
    hit1, hit3 = hit1 / cnt, hit3 / cnt

    rows_seen = np.concatenate([np.where(y == c)[0] for c in seen])
    rows_blank = np.where(y == n_cls - 1)[0]
    sep_blank = (np.median(dist[rows_blank].min(axis=1))
                 - np.median(dist[rows_seen].min(axis=1)))
    return {
        "hit@1_vs_doornb": float(hit1),
        "hit@3_vs_doornb": float(hit3),
        "chance_hit@1": 1.0 / len(seen),
        "n_seen": len(seen),
        "n_held": len(held),
        "median_blank_min_dist": float(np.median(dist[rows_blank].min(axis=1))),
        "median_seen_min_dist": float(np.median(dist[rows_seen].min(axis=1))),
        "blank_min_sep": float(sep_blank),
        "held_ids": [int(c) for c in held],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="decoder")
    args = ap.parse_args()
    t0 = time.perf_counter()

    d, names, meta, all_names, resp = load(args.outdir)
    print("loaded data", flush=True)
    y = d["y"]
    n_odor = int(meta["n_odors"])
    n_cls = len(names)
    reps = make_reps(d)
    print("built reps:", list(reps.keys()), flush=True)

    nd = {"n_cls": n_cls, "n_odors": n_odor, "chance_baseline": 1.0 / n_cls}
    out_path = os.path.join(args.outdir, "baseline.json")

    report = {}
    for rep, X in reps.items():
        print(f"> rep {rep} start", flush=True)
        cs = closed_set(X, y, n_cls)
        print(f"> rep {rep} closed done", flush=True)
        ho = heldout_metrics(X, y, n_odor, names, all_names, resp, n_cls)
        print(f"> rep {rep} heldout done", flush=True)
        report[rep] = {"closed": cs, "heldout": ho}
        with open(out_path, "w") as f:
            json.dump({"report": report, "note": nd}, f, indent=2)
        print(f"  (saved so far)", flush=True)
        print(f"\n=== {rep} (n_features={X.shape[1]}) ===", flush=True)
        for cname, r in cs.items():
            print(f"  closed-set {cname:10s}: acc@1={r['acc@1']:.3f} "
                  f"acc@3={r['acc@3']:.3f} blank_acc={r['blank_acc']:.3f}",
                  flush=True)
        print(f"  held-out (n={ho['n_held']}): hit@1={ho['hit@1_vs_doornb']:.3f} "
              f"hit@3={ho['hit@3_vs_doornb']:.3f} "
              f"(chance@1={ho['chance_hit@1']:.3f})  blank sep="
              f"{ho['blank_min_sep']:.1f} min-dist units", flush=True)

    with open(out_path, "w") as f:
        json.dump({"report": report, "note": nd}, f, indent=2)
    print(f"\nsaved {out_path}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()