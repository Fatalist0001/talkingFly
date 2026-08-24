"""Weight-transform sweep: does the ORN->AL transformation improve if we map
syn_count to LIF weight differently?

The baseline model uses the arbitrary scaling sign * max(0.1, syn_count).
This script rebuilds the SAME AL subgraph once, then re-simulates an identical
set of jittered odor trials under several weight transforms (see
run_odors.weight_transform) and compares:

  closed 1-NN LOO accuracy on window spike counts (main number),
  intra-trial reliability (mean corr between same-odor trials),
  min/mean pairwise corr between odor-mean states,
  activity sanity (active fraction of neurons).

Drives are pre-drawn ONCE and reused across transforms, so comparisons are
paired. Results go to results_weights/weights_sweep.json.

Usage:
  python sweep_weights.py --n-odors 10 --trials 4 --workers 8
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import build_odor_dataset as bd
import prepare_olfaction as op
from brian2 import amp, ms, pA, run as b2run
from run_odors import WEIGHT_TRANSFORMS, build_network, build_subgraph, \
    weight_transform

# ---- worker globals (set once per subprocess) ------------------------------
_ARGS = None
_N = None
_IS_ORN = None
_I_ARR = None
_J_ARR = None
_WT = None            # dict: transform name -> weight array


def _init_worker(args, n, is_orn, i_arr, j_arr, wt_map):
    global _ARGS, _N, _IS_ORN, _I_ARR, _J_ARR, _WT
    _ARGS, _N, _IS_ORN, _I_ARR, _J_ARR, _WT = \
        args, n, is_orn, i_arr, j_arr, wt_map


def _sim_one(w_syn, drive):
    """One tonic trial; returns per-neuron spike counts in the stim window."""
    a = _ARGS
    G, S, sm, Rres, El = build_network(_N, _I_ARR, _J_ARR, w_syn)
    G.v = El + a.base * pA * Rres
    G.I_drive[:] = a.base * pA
    G.I_drive[~_IS_ORN] = 0 * amp
    b2run(a.stim_start * ms)
    inj = np.zeros(_N)
    inj[_IS_ORN] = a.base + np.clip(drive, 0.0, None)[_IS_ORN]
    G.I_drive[:] = inj * pA
    b2run(a.stim_dur * ms)
    G.I_drive[:] = a.base * pA
    G.I_drive[~_IS_ORN] = 0 * amp
    b2run((a.simtime - a.stim_start - a.stim_dur) * ms)

    t = sm.t[:] / ms
    ii = sm.i[:]
    win = (t >= a.stim_start) & (t < a.stim_start + a.stim_dur)
    return np.bincount(ii[win], minlength=_N).astype(np.float32)


def _run_chunk(specs):
    return [(tn, label, _sim_one(_WT[tn], drive))
            for (tn, label, drive) in specs]


def evaluate(X, y, n_odors):
    X = X.astype(np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    acc = float(np.mean(y[np.argmax(sims, axis=1)] == y))

    rels = []
    for l in range(n_odors):
        vs = X[y == l]
        if len(vs) > 1:
            c = np.corrcoef(vs)[np.triu_indices(len(vs), 1)]
            rels.append(float(np.nanmean(c)))
    means = np.stack([X[y == l].mean(0) for l in range(n_odors)])
    c = np.corrcoef(means)
    off = c[np.triu_indices(n_odors, 1)]
    return {
        "closed_acc": acc,
        "reliability": float(np.mean(rels)) if rels else float("nan"),
        "min_state_corr": float(np.nanmin(off)),
        "mean_state_corr": float(np.nanmean(off)),
        "active_frac": float(np.mean(
            np.count_nonzero(X, axis=1) / max(1, X.shape[1]))),
        "zero_trials": int(np.sum(~np.any(X > 0, axis=1))),
        "max_count": float(X.max()),
        "spikes_mean": float(X.sum(1).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feather", default="proofread_connections_783.feather")
    p.add_argument("--neurons", type=int, default=1500, help="top-K AL presyn")
    p.add_argument("--simtime", type=float, default=150)
    p.add_argument("--stim-start", type=float, default=50)
    p.add_argument("--stim-dur", type=float, default=70)
    p.add_argument("--base", type=float, default=1.0)
    p.add_argument("--gain", type=float, default=40.0)
    p.add_argument("--drive-sigma", type=float, default=0.15)
    p.add_argument("--n-odors", type=int, default=10)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transforms", default=",".join(WEIGHT_TRANSFORMS))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--outdir", default="results_weights")
    args = p.parse_args()

    orn, matrix, names = op.load()
    orn_ids = orn["root_id"].astype(np.int64).to_numpy()
    odor_idx = bd.select_odors(matrix, names, args.n_odors, args.seed)
    print(f"odors ({len(odor_idx)}): "
          f"{', '.join(names[i] for i in odor_idx)}")

    chosen, i_arr, j_arr, w0, is_orn = build_subgraph(
        args.feather, args.neurons, orn_ids)
    N = len(chosen)
    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    wt = {t: weight_transform(w0, i_arr, j_arr, N, t) for t in transforms}

    # ---- paired drives: drawn once, reused by every transform ---------------
    rng = np.random.default_rng(args.seed)
    n_orn = int(is_orn.sum())
    drives = []
    for oi in odor_idx:
        v = np.where(np.isnan(matrix[:, oi]), 0.0, matrix[:, oi])
        for _ in range(args.trials):
            d = np.zeros(N)
            d[is_orn] = args.gain * v * (
                1.0 + args.drive_sigma * rng.standard_normal(n_orn))
            drives.append((oi, d))

    specs = [(tn, pi, d) for tn in transforms for pi, (_, d) in
             enumerate(drives)]
    n_workers = max(1, min(args.workers or 1, len(specs)))

    results = {t: [] for t in transforms}
    # drives are ordered odor-major (trials contiguous), so odor labels are
    # simply the repeated position range -- NOT the raw DoOR column ids.
    y_drives = np.repeat(np.arange(len(odor_idx)), args.trials)
    y_by_transform = {t: y_drives for t in transforms}
    t_begin = time.perf_counter()
    if n_workers > 1:
        chunks = bd._chunkify(specs, n_workers)
        done = 0
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(args, N, is_orn, i_arr, j_arr, wt)) as ex:
            futs = [ex.submit(_run_chunk, ch) for ch in chunks]
            for fut in as_completed(futs):
                for tn, label, counts in fut.result():
                    results[tn].append((label, counts))
                done += len(chunks[0])
                print(f"  {done}/{len(specs)} trials "
                      f"({time.perf_counter() - t_begin:.0f}s)", flush=True)
    else:
        _init_worker(args, N, is_orn, i_arr, j_arr, wt)
        for tn, label, counts in _run_chunk(specs):
            results[tn].append((label, counts))

    # ---- assemble & score -----------------------------------------------------
    rows = []
    for tn in transforms:
        pairs = sorted(results[tn], key=lambda x: x[0])
        X = np.stack([c for _, c in pairs])
        y = y_by_transform[tn]
        m = evaluate(X, y, len(odor_idx))
        w = wt[tn]
        m.update({
            "transform": tn,
            "w_absmax": float(np.abs(w).max()),
            "w_absmean": float(np.abs(w).mean()),
        })
        rows.append(m)
    rows.sort(key=lambda r: -r["closed_acc"])

    print(f"\n{'transform':10s} {'acc@1':>6s} {'rel':>5s} {'minC':>6s} "
          f"{'meanC':>6s} {'act%':>5s} {'zero':>4s} {'spk':>6s} {'|w|max':>7s}")
    chance = 1.0 / len(odor_idx)
    for r in rows:
        print(f"{r['transform']:10s} {r['closed_acc']:6.3f} "
              f"{r['reliability']:5.2f} {r['min_state_corr']:6.3f} "
              f"{r['mean_state_corr']:6.3f} {r['active_frac']*100:4.0f}% "
              f"{r['zero_trials']:4d} {r['spikes_mean']:6.0f} "
              f"{r['w_absmax']:7.1f}")
    print(f"(chance acc@1 = {chance:.3f})")

    report = {"args": vars(args),
              "odor_idx": [int(i) for i in odor_idx],
              "odor_names": [names[i] for i in odor_idx],
              "n_neurons": N, "results": rows}
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "weights_sweep.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {out}  wall={time.perf_counter() - t_begin:.0f}s")


if __name__ == "__main__":
    main()
