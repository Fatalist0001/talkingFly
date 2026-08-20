"""Stage 3: run the Brian2 network for different odors and compare brain states.

The subgraph is built around the antennal lobe (AL): top-K presynaptic neurons
by AL synapse degree plus ALL olfactory receptor neurons (ORNs) from the
FlyWire annotations. ORNs are the driven input: for each odor they receive
I_inj = base_pA + gain_pA * DoOR_response_to_that_odor.

For each odor we record spikes, compute per-neuron spike counts in the odor
window, and report how distinct the evoked states are (pairwise cosine).

Usage:
    python run_odors.py --neurons 2000 --simtime 200 --gain 4 \
        --odors "ethyl acetate,benzaldehyde,1-octanol,ethyl butyrate"
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.feather as ft
from brian2 import *

import prepare_olfaction as op


def build_network(n_neurons, i_arr, j_arr, w_syn):
    """Fresh deterministic Brian2 LIF network over the neuron set."""
    start_scope()
    defaultclock.dt = 0.1 * ms
    Cm = 1 * pF
    tau = 10 * ms
    gl = Cm / tau
    Rres = 1 / gl
    El = -70 * mV
    Vt = -50 * mV
    Vr = -65 * mV
    tau_syn = 2 * ms
    eqs = """
    dv/dt = (El - v)/tau + (Isyn + I_drive)*Rres/tau : volt
    dIsyn/dt = -Isyn/tau_syn : amp
    I_drive : amp
    """
    G = NeuronGroup(n_neurons, model=eqs, threshold="v > Vt",
                    reset="v = Vr", refractory=2 * ms, method="euler",
                    namespace={"tau": tau, "tau_syn": tau_syn,
                               "Rres": Rres, "El": El, "Vt": Vt, "Vr": Vr})
    G.I_drive = 0 * amp
    S = Synapses(G, G, model="weight : 1", on_pre="Isyn += weight*amp")
    S.connect(i=i_arr, j=j_arr)
    S.weight = w_syn
    sm = SpikeMonitor(G)
    return G, S, sm, Rres, El


def build_subgraph(feather, n_top_al, orn_ids):
    """Top-K AL presynaptic neurons + all ORNs; returns edge arrays."""
    t = ft.read_table(feather)
    pre = t.column("pre_pt_root_id").to_numpy()
    post = t.column("post_pt_root_id").to_numpy()
    npl = t.column("neuropil").to_pylist()
    syn_c = t.column("syn_count").to_numpy().astype(np.float64)
    gaba = t.column("gaba_avg").to_numpy()
    glut = t.column("glut_avg").to_numpy()

    in_al = np.array([isinstance(x, str) and x.startswith("AL") for x in npl])
    pre_a = pre[in_al]
    uniq, counts = np.unique(pre_a, return_counts=True)
    top = uniq[np.argsort(-counts)[: n_top_al]]
    chosen = np.unique(np.concatenate([top, orn_ids]))
    cset = set(chosen.tolist())
    print(f"chosen neurons: {len(chosen)} "
          f"(ORNs among them: {len(cset & set(orn_ids.tolist()))})")

    keep = np.isin(pre, list(cset)) & np.isin(post, list(cset))
    pre_k, post_k = pre[keep], post[keep]
    id2i = {nid: i for i, nid in enumerate(chosen)}
    i_arr = np.array([id2i[a] for a in pre_k], dtype=np.int64)
    j_arr = np.array([id2i[b] for b in post_k], dtype=np.int64)
    syn_k = syn_c[keep]
    inhibitory = gaba[keep] > glut[keep]
    w_syn = np.where(inhibitory, -1.0, 1.0) * np.maximum(0.1, syn_k)
    print(f"synapses in subgraph: {len(i_arr)} "
          f"(E: {int((~inhibitory).sum())}, I: {int(inhibitory.sum())})")
    orn_set = set(orn_ids.tolist())
    is_orn = np.array([n in orn_set for n in chosen], dtype=bool)
    return chosen, i_arr, j_arr, w_syn, is_orn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feather", default="proofread_connections_783.feather")
    p.add_argument("--neurons", type=int, default=2000, help="top-K AL presyn")
    p.add_argument("--simtime", type=float, default=200, help="ms, total")
    p.add_argument("--stim-start", type=float, default=50, help="ms, odor onset")
    p.add_argument("--stim-dur", type=float, default=100, help="ms, odor duration")
    p.add_argument("--base", type=float, default=1.0, help="pA tonic ORN drive")
    p.add_argument("--gain", type=float, default=5.0,
                   help="pA per unit DoOR response (0..1)")
    p.add_argument("--odors",
                   default="ethyl acetate,benzaldehyde,1-octanol,"
                           "ethyl butyrate,citronellal,isobutyl acetate",
                   help="comma-separated odor names")
    p.add_argument("--outdir", default="results_odor")
    args = p.parse_args()

    orn, matrix, names = op.load()
    orn_ids = orn["root_id"].astype(np.int64).to_numpy()
    orn_tab_idx = {oid: i for i, oid in enumerate(orn_ids)}
    print(f"ORNs: {len(orn_ids)}, odors in matrix: {matrix.shape[1]}")

    # ---- subgraph: top-K presynaptic in AL + all ORNs ------------------------
    chosen, i_arr, j_arr, w_syn, is_orn = build_subgraph(
        args.feather, args.neurons, orn_ids)

    # ---- odor list -----------------------------------------------------------
    odor_names = [s.strip() for s in args.odors.split(",") if s.strip()]
    odor_idx = [op.find_odor(nm, names) for nm in odor_names]

    # ---- run trials ----------------------------------------------------------
    all_states = []
    file_rows = []
    t_begin = time.perf_counter()

    for rank, stim_i in enumerate(odor_idx + [-1]):  # -1 = blank
        ev = op.stimulus(stim_i, gain_pA=args.gain, base_pA=args.base) \
            if stim_i >= 0 else None
        G, S, sm, Rres, El = build_network(len(chosen), i_arr, j_arr, w_syn)
        G.v = El + args.base * pA * Rres
        G.I_drive[:] = args.base * pA
        G.I_drive[~is_orn] = 0 * amp

        t0 = time.perf_counter()
        if ev is not None:
            run(args.stim_start * ms)
            cur = dict(zip(ev["root_id"].astype(np.int64),
                           ev["I_inj_pA"], strict=False))
            inj = np.zeros(len(chosen))
            inj[is_orn] = [cur[n] for n in chosen[is_orn]]
            G.I_drive[:] = inj * pA
            run(args.stim_dur * ms)
            G.I_drive[:] = args.base * pA
            G.I_drive[~is_orn] = 0 * amp
            run((args.simtime - args.stim_start - args.stim_dur) * ms)
        else:
            run(args.simtime * ms)
        wall = time.perf_counter() - t0

        win = (sm.t[:] / ms >= args.stim_start) & \
              (sm.t[:] / ms < args.stim_start + args.stim_dur)
        win_counts = np.bincount(sm.i[:][win], minlength=len(chosen))
        all_states.append(win_counts.astype(np.float64))

        os.makedirs(args.outdir, exist_ok=True)
        np.save(os.path.join(args.outdir, f"spikes_{rank:02d}_t.npy"),
                sm.t[:] / ms)
        np.save(os.path.join(args.outdir, f"spikes_{rank:02d}_i.npy"), sm.i[:])

        name = odor_names[rank] if rank < len(odor_names) else "__blank__"
        print(f"[{rank}] {name:22s} idx={stim_i}  wall={wall:4.1f}s  "
              f"spikes_in_window={win_counts.sum():7d}  "
              f"active={int(np.count_nonzero(win_counts)):5d}")
        file_rows.append({"trial": rank, "odor": name, "odor_index": stim_i,
                          "spikes_in_window": int(win_counts.sum()),
                          "active_in_window": int(np.count_nonzero(win_counts))})

    all_states = np.array(all_states)
    np.save(os.path.join(args.outdir, "states.npy"), all_states)
    orn_ct = dict(zip(orn_ids, orn["cell_type"], strict=False))
    pd.DataFrame({
        "local_index": np.arange(len(chosen)),
        "root_id": chosen,
        "is_orn": is_orn,
        "cell_type": [orn_ct.get(int(n), "") for n in chosen],
    }).to_csv(os.path.join(args.outdir, "neurons.csv"), index=False)

    # ---- discriminability -----------------------------------------------------
    blank = all_states[-1]
    states = all_states[:-1]

    def cos(a, b, eps=1e-8):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

    # remove blank spike-window from discrimination if blank silent
    blank_active = bool(np.count_nonzero(blank))

    metrics = {"n_neurons": int(len(chosen)), "n_orns": int(is_orn.sum()),
               "n_odors": int(len(states)), "simtime_ms": args.simtime,
               "stim_window_ms": args.stim_dur,
               "blank_spikes_in_window": int(blank.sum()),
               "blank_active": blank_active}
    for subset, msk in [("all", np.ones(len(chosen), bool)),
                        ("orn", is_orn),
                        ("non_orn", ~is_orn)]:
        S, B = states[:, msk], blank[msk]
        sim_pairs = [cos(s, tt) for a, s in enumerate(S) for tt in S[a + 1:]]
        sim_blank = [cos(s, B) for s in S] if blank_active else []
        metrics[subset] = {
            "mean_pairwise_cos": float(np.mean(sim_pairs)) if sim_pairs else 1.0,
            "min_pairwise_cos": float(np.min(sim_pairs)) if sim_pairs else 1.0,
            "mean_cos_to_blank": float(np.mean(sim_blank))
                if sim_blank else 0.0,
            "max_cos_to_blank": float(np.max(sim_blank))
                if sim_blank else 0.0,
            "mean_state_norm": float(np.mean(np.linalg.norm(S, axis=1))),
            "fraction_active": float(np.mean(
                np.count_nonzero(S, axis=1) / max(1, int(msk.sum())))),
        }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({"metrics": metrics, "trials": file_rows,
                   "total_wall_s": round(time.perf_counter() - t_begin, 2)},
                  f, indent=2)

    print("\n=== discriminability (odor window) ===")
    print(f"blank: {int(blank.sum())} spikes "
          f"({'active' if blank_active else 'SILENT (no spontaneous drive)'})")
    for sub, m in metrics.items():
        if not isinstance(m, dict):
            continue
        print(f"[{sub:7s}] mean pairwise cos = {m['mean_pairwise_cos']:.3f} "
              f"(min {m['min_pairwise_cos']:.3f})  |  cos to blank = "
              f"{m['mean_cos_to_blank']:.3f} (max {m['max_cos_to_blank']:.3f}) "
              f"| active frac = {m['fraction_active']:.2f}")
    print(f"\nsaved to {args.outdir}/  (states.npy, spikes_*.npy, "
          f"neurons.csv, summary.json)")


if __name__ == "__main__":
    main()