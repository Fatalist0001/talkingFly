"""
FlyWire connectome -> Brian2 -> Brian2CUDA pipeline.
Loads an extracted subgraph from proofread_connections_783.feather,
builds a LIF network in Brian2 and runs it on an NVIDIA GPU.

Usage:
    python flynet.py --neuropil EB --neurons 2000 --duration 200 --simtime 200

Model description:
    Leaky Integrate-and-Fire (LIF):
      tau*dv/dt = -(v - El) + R*Isyn
      if v > Vt: spike, v -> Vr
    Current-based exponential synapses:
      tau_syn*dIsyn/dt = -Isyn
    Neurons are classified excitatory (glutamatergic/cholinergic) or
    inhibitory (GABAergic) from the connectome's transmitter-averaging.
    Connection strength is scaled by syn_count.
"""
import argparse
import time

import numpy as np
import pyarrow.feather as ft
from brian2 import *
# For GPU (WSL2/Linux) add: import brian2cuda; set_device("cuda_standalone")

parser = argparse.ArgumentParser()
parser.add_argument("--feather", default="proofread_connections_783.feather")
parser.add_argument("--neuropil", default="EB", help="brain region to extract")
parser.add_argument("--neurons", type=int, default=2000, help="top-K presynaptic neurons by degree")
parser.add_argument("--simtime", type=float, default=200, help="simulation time in ms")
parser.add_argument("--dt", type=float, default=0.1, help="time step in ms")
args = parser.parse_args()

print("Loading connectome ...")
table = ft.read_table(args.feather)
pre_arr = table.column("pre_pt_root_id").to_numpy()
post_arr = table.column("post_pt_root_id").to_numpy()
neuropil_arr = table.column("neuropil").to_numpy()
syn_count = table.column("syn_count").to_numpy().astype(np.float64)
gaba = table.column("gaba_avg").to_numpy()
glut = table.column("glut_avg").to_numpy()

mask = neuropil_arr == args.neuropil
print(f"connections in {args.neuropil}: {mask.sum()}")

pre_p = pre_arr[mask]
post_p = post_arr[mask]
syn_p = syn_count[mask]
gaba_p = gaba[mask]
glut_p = glut[mask]

uniq, counts = np.unique(pre_p, return_counts=True)
top = np.argsort(-counts)[: args.neurons]
chosen = np.unique(uniq[top])
if len(chosen) < args.neurons:
    chosen = np.unique(pre_p)
    print(f"only {len(chosen)} unique presynaptic neurons present")
print(f"selected {len(chosen)} neurons")

id2idx = {nid: i for i, nid in enumerate(chosen)}

# collect post-synaptic targets that are also in chosen set
sel_pre = []
sel_post = []
sel_w = []
sel_ex = []
for k in range(len(pre_p)):
    i = id2idx.get(pre_p[k])
    if i is None:
        continue
    j = id2idx.get(post_p[k])
    if j is None:
        continue
    # classify pre-synaptic neuron by its transmitter profile
    inhibitory = gaba_p[k] > glut_p[k]
    sign = -1.0 if inhibitory else 1.0
    w = sign * max(0.1, syn_p[k])  # weight in arbitrary units
    sel_pre.append(i)
    sel_post.append(j)
    sel_w.append(w)
    sel_ex.append(0 if inhibitory else 1)

sel_pre = np.asarray(sel_pre)
sel_post = np.asarray(sel_post)
w_syn = np.asarray(sel_w)
n_ex = np.sum(np.asarray(sel_ex))
print(f"synapses in subgraph: {len(sel_pre)}  (excitatory: {n_ex})")

# ---------------------------------------------------------------------------
# Brian2 model
# ---------------------------------------------------------------------------
start_scope()
defaultclock.dt = args.dt * ms

Cm = 1 * pF
tau = 10 * ms
gl = Cm / tau
R = 1 / gl
El = -70 * mV
Vt = -50 * mV
Vr = -65 * mV
tau_syn = 2 * ms

eqs = """
dv/dt = (El - v)/tau + (Isyn + I_drive)*R/tau : volt
dIsyn/dt = -Isyn/tau_syn : amp
I_drive : amp
"""

G = NeuronGroup(
    len(chosen),
    model=eqs,
    threshold="v > Vt",
    reset="v = Vr",
    refractory=2 * ms,
    method="euler",
    name="neurons",
)

# drive a third of the neurons with a constant current so the net is active
rng = np.random.default_rng(42)
driven = rng.choice(len(chosen), size=len(chosen) // 3, replace=False)
G.I_drive = 0 * amp
G.I_drive[driven] = 3.5 * pA
G.v = El + G.I_drive * R

S = Synapses(
    G,
    G,
    model="weight : 1",
    on_pre="Isyn += weight*amp",
    name="synapses",
)
S.connect(i=sel_pre, j=sel_post)
S.weight = w_syn

spikemon = SpikeMonitor(G)

print(f"running {args.simtime} ms on CPU (numpy device) ...")
t0 = time.perf_counter()
run(args.simtime * ms)
wall = time.perf_counter() - t0

n_spikes = spikemon.count[:]
n_in = np.sum(w_syn < 0)
print(f"\n=== results ===")
print(f"simulated {args.simtime} ms in {wall:.2f} s wall time")
print(f"neurons: {len(chosen)}  | synapses: {len(sel_pre)} "
      f"(E: {n_ex}, I: {n_in})")
print(f"spiking neurons: {np.count_nonzero(n_spikes)}/{len(chosen)} "
      f"({100 * np.count_nonzero(n_spikes) / len(chosen):.1f}%)")
total = sum(n_spikes)
print(f"total spikes: {total} | mean rate (spiking pool): "
      f"{total / max(1, np.count_nonzero(n_spikes)) / (args.simtime / 1000.0):.2f} Hz")

import json
with open("flynet_result.json", "w") as f:
    json.dump({
        "neuropil": args.neuropil,
        "neurons": int(len(chosen)),
        "synapses": int(len(sel_pre)),
        "excitatory_synapses": int(n_ex),
        "inhibitory_synapses": int(n_in),
        "spiking_neurons": int(np.count_nonzero(n_spikes)),
        "total_spikes": int(total),
        "wall_time_s": round(wall, 3),
    }, f, indent=2)
print("saved flynet_result.json")