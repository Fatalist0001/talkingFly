"""Stage 4.1: build a decoder training dataset of odor-evoked brain states.

Repeated trials per odor with random stimulus jitter (so each trial differs),
many odors, and several feature representations:
  - X_counts       : per-neuron spike counts in the stimulus window  (T, N)
  - X_bins         : per-neuron binned counts across the window      (T, B, N)
  - X_glom         : counts aggregated over ORN glomeruli + "other"   (T, D)
  - X_glom_bins    : binned glom aggregates                          (T, B, D)
plus labels y, neuron/glom metadata and a params json.

Odor selection samples a chemically diverse subset of the 691 DoOR odors
(greedy: drop candidates too similar to already-kept ones).

Trials are generated in parallel across CPU cores (numpy device).  Each trial
is simulated in its own fresh network, so the dataset is bit-for-bit the same
as a serial run (drive vectors are pre-drawn in the parent with a fixed seed).

Usage:
  python build_odor_dataset.py --n-odors 30 --trials 20 --drive-sigma 0.15
  python build_odor_dataset.py --workers 8 --n-odors 100 --trials 100
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from brian2 import *

import prepare_olfaction as op
from run_odors import build_network, build_subgraph


# ---- worker globals (set once per subprocess) ------------------------------
_ARGS = None
_N = None
_IS_ORN = None
_I_ARR = None
_J_ARR = None
_W_SYN = None


def _init_worker(args, N, is_orn, i_arr, j_arr, w_syn):
    global _ARGS, _N, _IS_ORN, _I_ARR, _J_ARR, _W_SYN
    _ARGS, _N, _IS_ORN = args, N, is_orn
    _I_ARR, _J_ARR, _W_SYN = i_arr, j_arr, w_syn


def _chunkify(specs, n):
    n = max(1, min(n, len(specs)))
    k = (len(specs) + n - 1) // n
    return [specs[i:i + k] for i in range(0, len(specs), k)]


def _run_chunk(specs):
    out = []
    for (label, kind, oi, drive_orn) in specs:
        counts, bins, traj = simulate_trial(
            drive_orn, _ARGS, _N, _IS_ORN, _I_ARR, _J_ARR, _W_SYN)
        out.append((counts, bins, traj, label, kind, oi))
    return out


def simulate_trial(drive_orn, args, N, is_orn, i_arr, j_arr, w_syn):
    """Simulate one trial. drive_orn: (N,) pA drive (non-ORN set to 0)."""
    G, S, sm, Rres, El = build_network(N, i_arr, j_arr, w_syn)
    G.v = El + args.base * pA * Rres
    G.I_drive[:] = args.base * pA
    G.I_drive[~is_orn] = 0 * amp
    run(args.stim_start * ms)
    d = np.zeros(N)
    d[is_orn] = np.clip(drive_orn[is_orn], 0.0, None)
    G.I_drive[:] = d * pA
    run(args.stim_dur * ms)
    G.I_drive[:] = args.base * pA
    G.I_drive[~is_orn] = 0 * amp
    run((args.simtime - args.stim_start - args.stim_dur) * ms)

    t_all = sm.t[:] / ms
    i_all = sm.i[:]

    # ---- stimulus-window features (as before) --------------------------------
    win = (t_all >= args.stim_start) & (t_all < args.stim_start + args.stim_dur)
    tw, iw = t_all[win], i_all[win]
    B = args.nbins
    binw = args.stim_dur / B
    counts = np.bincount(iw, minlength=N).astype(np.float32)
    bind = np.clip(np.searchsorted(
        np.arange(B + 1) * binw + args.stim_start, tw, side="right") - 1,
        0, B - 1)
    flat = bind * N + iw
    bins = np.bincount(flat, minlength=B * N).astype(np.float32).reshape(B, N)

    # ---- full-window temporal trajectory (lever 2) ---------------------------
    TB = args.traj_bins
    edges = np.linspace(0.0, args.simtime, TB + 1)
    tb = np.clip(np.searchsorted(edges, t_all, side="right") - 1, 0, TB - 1)
    traj = np.bincount(tb * N + i_all, minlength=TB * N).astype(np.float32) \
        .reshape(TB, N)
    return counts, bins, traj


def select_odors(matrix, names, n_want, seed, max_corr=0.9,
                min_active=4, resp_thr=0.3):
    """Pick n_want diverse, receptor-activating odor columns.

    A candidate must excite at least `min_active` of our receptor columns at
    response >= resp_thr (otherwise its brain state is identical to blank and
    it is undecodable). Greedy diversity: drop candidates too similar to
    already-kept ones.
    """
    rng = np.random.default_rng(seed)
    clean = np.nan_to_num(matrix, nan=0.0)
    valid = []
    for i in range(matrix.shape[1]):
        v = clean[:, i]
        if np.all(v == v[0]):          # constant / all-zero
            continue
        if (v >= resp_thr).sum() < min_active:  # wouldn't drive our ORNs
            continue
        valid.append(i)
    kept = []
    order = rng.permutation(valid)
    for i in order:
        v = clean[:, i]
        if any(np.corrcoef(v, clean[:, j])[0, 1] > max_corr
               for j in kept):
            continue
        kept.append(i)
        if len(kept) >= n_want:
            break
    if len(kept) < n_want:
        kept = list(order[:n_want])
        kept.sort()
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feather", default="proofread_connections_783.feather")
    p.add_argument("--neurons", type=int, default=1500, help="top-K AL presyn")
    p.add_argument("--simtime", type=float, default=150, help="ms, total")
    p.add_argument("--stim-start", type=float, default=50, help="ms, odor onset")
    p.add_argument("--stim-dur", type=float, default=70, help="ms, odor duration")
    p.add_argument("--nbins", type=int, default=7, help="PSTH bins in window")
    p.add_argument("--traj-bins", type=int, default=30,
                   help="full-window PSTH bins (temporal trajectory feature)")
    p.add_argument("--base", type=float, default=1.0, help="pA tonic ORN drive")
    p.add_argument("--gain", type=float, default=5.0, help="pA per resp unit")
    p.add_argument("--drive-sigma", type=float, default=0.15,
                   help="relative trial-to-trial jitter of ORN drive")
    p.add_argument("--odors", default=None, help="comma-separated names")
    p.add_argument("--n-odors", type=int, default=30)
    p.add_argument("--trials", type=int, default=20, help="trials per odor")
    p.add_argument("--n-blank", type=int, default=10, help="blank trials")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="parallel CPU workers (0/1 = serial)")
    p.add_argument("--outdir", default="decoder")
    args = p.parse_args()

    orn, matrix, names = op.load()
    orn_ids = orn["root_id"].astype(np.int64).to_numpy()
    orn_ct = dict(zip(orn_ids, orn["cell_type"], strict=False))

    if args.odors:
        odor_idx = [op.find_odor(s.strip(), names) for s in
                    args.odors.split(",") if s.strip()]
        odor_idx = [i for i in odor_idx if i >= 0]
        print(f"explicit odors: {len(odor_idx)}")
    else:
        odor_idx = select_odors(matrix, names, args.n_odors, args.seed)
        print(f"sampled {len(odor_idx)} diverse odors (seed={args.seed})")

    chosen, i_arr, j_arr, w_syn, is_orn = build_subgraph(
        args.feather, args.neurons, orn_ids)
    N = len(chosen)
    rng = np.random.default_rng(args.seed)

    # ---- glomerulus grouping matrix (N, D) ------------------------------------
    types = sorted({orn_ct.get(int(n), "") for n in chosen})
    types = [t for t in types if t.startswith("ORN_")] + ["other"]
    type_id = {t: i for i, t in enumerate(types)}
    D = len(types)
    group_ids = np.array([type_id[orn_ct.get(int(n), "other")] for n in chosen])
    onehot = np.zeros((N, D), dtype=np.float32)
    onehot[np.arange(N), group_ids] = 1.0

    # ---- pre-draw every trial's drive vector (fixed seed -> reproducible) -----
    t_begin = time.perf_counter()
    n_orn = int(is_orn.sum())
    trials_spec = []
    for pi, oi in enumerate(odor_idx):          # pi = label of this odor
        resp = matrix[:, oi]
        v = np.where(np.isnan(resp), 0.0, resp)
        for _ in range(args.trials):
            d = np.zeros(N)
            d[is_orn] = args.base + args.gain * v * (
                1.0 + args.drive_sigma * rng.standard_normal(n_orn))
            trials_spec.append((pi, "odor", oi, d))
    for _ in range(args.n_blank):
        d = np.zeros(N)
        d[is_orn] = args.base * (
            1.0 + args.drive_sigma * rng.standard_normal(n_orn))
        trials_spec.append((len(odor_idx), "blank", -1, d))
    n_total = len(trials_spec)

    # ---- run trials (parallel) ------------------------------------------------
    n_workers = 1 if (args.workers or 0) <= 1 else args.workers
    if n_workers > 1 and n_total > 1:
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(args, N, is_orn, i_arr, j_arr, w_syn)) as ex:
            chunks = _chunkify(trials_spec, n_workers)
            results = list(ex.map(_run_chunk, chunks))
    else:
        _init_worker(args, N, is_orn, i_arr, j_arr, w_syn)
        results = [_run_chunk(trials_spec)]
    print(f"built {n_total}/{n_total} trials in "
          f"{time.perf_counter() - t_begin:.1f}s "
          f"(workers={n_workers})", flush=True)

    # ---- assemble -------------------------------------------------------------
    X_counts, X_bins, X_traj, y, trials_meta = [], [], [], [], []
    for chunk in results:
        for (counts, bins, traj, label, kind, oi) in chunk:
            X_counts.append(counts)
            X_bins.append(bins)
            X_traj.append(traj)
            y.append(label)
            trials_meta.append({"kind": kind, "odor_index": int(oi)})
    X_counts = np.array(X_counts, dtype=np.float32)
    X_bins = np.array(X_bins, dtype=np.float32)
    X_traj = np.array(X_traj, dtype=np.float32)
    X_glom = X_counts @ onehot
    X_glom_bins = X_bins @ onehot
    X_glom_traj = X_traj @ onehot
    y = np.array(y, dtype=np.int64)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "dataset.npz"),
             X_counts=X_counts, X_bins=X_bins, X_traj=X_traj,
             X_glom=X_glom, X_glom_bins=X_glom_bins,
             X_glom_traj=X_glom_traj, y=y,
             root_ids=chosen.astype(np.int64), is_orn=is_orn,
             group_ids=group_ids)
    with open(os.path.join(args.outdir, "odor_names.json"), "w") as f:
        json.dump([names[i] for i in odor_idx] + ["__blank__"], f)
    pd_meta = {
        "n_neurons": N, "n_orns": int(is_orn.sum()), "n_odors": len(odor_idx),
        "trials_per_odor": args.trials, "n_blank": args.n_blank,
        "nbins": args.nbins, "traj_bins": args.traj_bins,
        "drive_sigma": args.drive_sigma,
        "base_pA": args.base, "gain_pA": args.gain,
        "simtime_ms": args.simtime, "stim_start_ms": args.stim_start,
        "stim_dur_ms": args.stim_dur, "glom_names": types,
        "workers": n_workers,
        "wall_s": round(time.perf_counter() - t_begin, 1),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(pd_meta, f, indent=2)
    print("saved:", os.path.join(args.outdir, "dataset.npz"),
          os.path.join(args.outdir, "meta.json"))

    # ---- sanity: per-odor reproducibility & diversity -------------------------
    odor_mask = y < len(odor_idx)
    Xo = X_counts[odor_mask]
    lab_o = y[odor_mask]
    means = np.stack([Xo[lab_o == l].mean(0) for l in range(len(odor_idx))])
    print("\nper-odor reproducibility (intra-trial correlation):")
    for l in range(len(odor_idx)):
        vs = Xo[lab_o == l]
        rep = float(np.corrcoef(vs)[np.triu_indices(len(vs), 1)].mean()) \
            if len(vs) > 1 else float("nan")
        print(f"  {names[odor_idx[l]]:26s} trials={len(vs)} "
              f"intra-corr={rep:.2f} norm={np.linalg.norm(means[l]):.1f}")
    c = np.corrcoef(means)
    off = np.abs(c[np.triu_indices(len(means), 1)])
    print(f"\nmin pairwise corr of mean states (odor vs odor): {off.min():.3f}")


if __name__ == "__main__":
    main()