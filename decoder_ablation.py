"""Stage 4.3 - ablations: how much of the decode comes from ORN vs downstream.

Same closed-set protocol as decoder_baseline, but restricted to neuron
subsets:
  - all      : every neuron (counts)
  - orn      : only ORNs (counts on ORN indices)
  - nonorn   : only non-ORN neurons (PNs / interneurons)
  - glom     : glomerulus-aggregated (ORN-only by construction)

This tests whether the downstream AL circuitry adds anything beyond the
(raw injected) ORN signal.

Usage:
  .\\.venv\\Scripts\\python.exe decoder_ablation.py [--outdir decoder]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 0
N_SPLITS = 3
TEST_FRAC = 0.25


def load(outdir):
    d = np.load(os.path.join(outdir, "dataset.npz"))
    return d, json.load(open(os.path.join(outdir, "meta.json")))


def closed_set(X, y, n_cls):
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=SEED)
    out = {}
    for cname, clf in {
        "logreg": LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        "linear_svm": SGDClassifier(
            loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-4,
            random_state=SEED),
    }.items():
        acc1, acc3, blank_acc = [], [], []
        for tr, te in sss.split(X, y):
            scl = StandardScaler().fit(X[tr])
            Xtr, Xte = scl.transform(X[tr]), scl.transform(X[te])
            m = clf.__class__(**clf.get_params())
            m.fit(Xtr, y[tr])
            scores = (m.predict_proba(Xte) if hasattr(m, "predict_proba")
                      else m.decision_function(Xte))
            order = np.argsort(-scores, axis=1)
            acc1.append((order[:, 0] == y[te]).mean())
            acc3.append((order[:, :3] == y[te][:, None]).any(axis=1).mean())
            bt = y[te] == n_cls - 1
            if bt.any():
                blank_acc.append((order[bt, 0] == n_cls - 1).mean())
        out[cname] = {
            "acc@1": float(np.mean(acc1)),
            "acc@3": float(np.mean(acc3)),
            "blank_acc": float(np.mean(blank_acc)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="decoder")
    args = ap.parse_args()
    t0 = time.perf_counter()

    d, meta = load(args.outdir)
    y = d["y"]
    n_cls = len(json.load(open(os.path.join(args.outdir, "odor_names.json"))))
    is_orn = d["is_orn"]
    counts = d["X_counts"].astype(np.float64)
    glom = d["X_glom"].astype(np.float64)

    subsets = {
        "all": counts,
        "orn": counts[:, is_orn],
        "nonorn": counts[:, ~is_orn],
        "glom": glom,
    }
    report = {}
    for name, X in subsets.items():
        cs = closed_set(X, y, n_cls)
        report[name] = {"n_features": int(X.shape[1]), "closed": cs}
        print(f"\n=== {name} (n_features={X.shape[1]}) ===", flush=True)
        for cname, r in cs.items():
            print(f"  {cname:10s}: acc@1={r['acc@1']:.3f} "
                  f"acc@3={r['acc@3']:.3f} blank_acc={r['blank_acc']:.3f}",
                  flush=True)

    out_path = os.path.join(args.outdir, "ablation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out_path}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()