"""Stages 4.4 + 4.5 - neural-net decoder and the brain->embedding bridge.

4.4 closed-set : MLP classifier among known odors (baseline comparison).
4.4 open-set   : regress the brain state to a compact ODOR EMBEDDING (a PCA
                  reduction of the DoOR receptor-response vector).  For held-out
                  (unseen) odors we predict that embedding and measure whether
                  it (a) lands near the chemically-similar seen odor and
                  (b) matches the TRUE unseen odor's embedding -> genuine
                  open-set generalization, not memorisation.
4.5 bridge      : the learned brain->embedding map IS the bridge to an LLM.
                  The target embedding is the olfactory "meaning" of each
                  odor.  If an external text/LLM embedding matrix exists at
                  decoder/text_embeddings.npy (shape [n_odors, dim], aligned
                  with odor_names.json) it is used INSTEAD of the DoOR PCA
                  space, with identical evaluation.

Usage:
  .\\.venv\\Scripts\\python.exe decoder_open.py [--outdir decoder]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 0
N_HELD = 9
N_SPLITS = 3
TEST_FRAC = 0.25
EMB_DIM = 30


def load(outdir):
    d = np.load(os.path.join(outdir, "dataset.npz"))
    names = json.load(open(os.path.join(outdir, "odor_names.json")))
    meta = json.load(open(os.path.join(outdir, "meta.json")))
    all_names = json.load(open("olfactory/odor_names.json"))
    resp = np.load("olfactory/resp_matrix.npy")
    return d, names, meta, all_names, resp


def corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def closed_mlp(X, y, n_cls):
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=SEED)
    acc1, acc3, blank_acc = [], [], []
    for tr, te in sss.split(X, y):
        scl = StandardScaler().fit(X[tr])
        Xtr, Xte = scl.transform(X[tr]), scl.transform(X[te])
        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                          random_state=SEED, early_stopping=True)
        m.fit(Xtr, y[tr])
        order = np.argsort(-m.predict_proba(Xte), axis=1)
        acc1.append((order[:, 0] == y[te]).mean())
        acc3.append((order[:, :3] == y[te][:, None]).any(axis=1).mean())
        bt = y[te] == n_cls - 1
        if bt.any():
            blank_acc.append((order[bt, 0] == n_cls - 1).mean())
    return {"acc@1": float(np.mean(acc1)),
            "acc@3": float(np.mean(acc3)),
            "blank_acc": float(np.mean(blank_acc))}


def open_set(X, y, n_odor, names, all_names, resp, emb_target, n_cls,
             regressor):
    """Train brain->embedding on SEEN odors; probe held-out odors."""
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(n_odor, N_HELD, replace=False))
    seen = np.array([c for c in range(n_odor) if c not in held])

    # expected DoOR-nearest seen odor for each held-out odor
    R = np.nan_to_num(resp, nan=0.0).astype(np.float64)
    col = [all_names.index(names[c]) for c in range(n_odor)]
    Rsel = R[:, col]                                  # (n_orn, n_odor)
    exp_nb = {}
    for c in held:
        inter = np.argsort([-corr2(Rsel[:, c], Rsel[:, s]) for s in seen])[0]
        exp_nb[int(c)] = int(seen[inter])

    scl = StandardScaler().fit(X)
    Z = scl.transform(X)
    tr_mask = np.isin(y, seen)
    Xtr, Ytr = Z[tr_mask], emb_target[seen].copy()
    # map each training trial to its odor's embedding
    Ytr_rows = emb_target[y[tr_mask]]
    m = regressor()
    m.fit(Xtr, Ytr_rows)

    # predictions per held-out trial
    pred = m.predict(Z[np.isin(y, held)])             # (n_held_trials, EMB)
    ho_emb = emb_target[held]                          # true held-out emb
    true_by_trial = emb_target[y[np.isin(y, held)]]

    # (a) retrieval vs seen odors via PREDICTED embedding
    seen_emb = emb_target[seen]
    hit1 = hit3 = cnt = 0
    for i, t in enumerate(true_by_trial):
        cos = np.array([corr2(pred[i], seen_emb[s]) for s in range(len(seen))])
        rk = np.argsort(-cos)
        exp = exp_nb[int(y[np.isin(y, held)][i])]
        hit1 += (rk[0] == exp)
        hit3 += (exp in rk[:3])
        cnt += 1
    # (b) true generalization: predicted vs TRUE held-out embedding
    cos_true = [corr2(pred[i], true_by_trial[i]) for i in range(len(pred))]
    # ceiling: true held-out emb vs nearest SEEN emb
    ceil = [max(corr2(ho_emb[c], seen_emb[s]) for s in range(len(seen)))
            for c in range(len(held))]

    return {
        "retrieval_hit@1_vs_doornb": hit1 / cnt,
        "retrieval_hit@3_vs_doornb": hit3 / cnt,
        "chance_hit@1": 1.0 / len(seen),
        "pred_cos_to_true_heldout_mean": float(np.mean(cos_true)),
        "pred_cos_to_true_heldout_std": float(np.std(cos_true)),
        "ceiling_true_vs_nearest_seen_mean": float(np.mean(ceil)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="decoder")
    args = ap.parse_args()
    t0 = time.perf_counter()

    d, names, meta, all_names, resp = load(args.outdir)
    y = d["y"]
    n_odor = int(meta["n_odors"])
    n_cls = len(names)
    counts = d["X_counts"].astype(np.float64)

    # ---- embedding target (DoOR PCA, or external text embedding) -------------
    emb_path = os.path.join(args.outdir, "text_embeddings.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path).astype(np.float64)
        print(f"using EXTERNAL text embedding ({emb.shape})", flush=True)
    else:
        R = np.nan_to_num(resp, nan=0.0).astype(np.float64)
        col = [all_names.index(names[c]) for c in range(n_odor)]
        Rsel = R[:, col].T                                    # (n_odor, n_orn)
        pca = PCA(n_components=EMB_DIM, random_state=SEED)
        emb = pca.fit_transform(Rsel)                         # (n_odor, EMB)
        print(f"using DoOR-PCA embedding dim={EMB_DIM} "
              f"(explained var {pca.explained_variance_ratio_.sum():.2f})",
              flush=True)

    report = {}
    print("closed-set MLP (counts)...", flush=True)
    report["closed_mlp"] = closed_mlp(counts, y, n_cls)
    print(f"  {report['closed_mlp']}", flush=True)

    for rname, reg in {
        "ridge": lambda: Ridge(alpha=1.0),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=400,
            random_state=SEED, early_stopping=True),
    }.items():
        res = open_set(counts, y, n_odor, names, all_names, resp, emb,
                       n_cls, reg)
        report[f"open_{rname}"] = res
        print(f"  open_{rname}: {res}", flush=True)

    out_path = os.path.join(args.outdir, "open.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out_path}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()