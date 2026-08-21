# AGENTS.md

Simulation pipeline for a Drosophila connectome (FlyWire FAFB v783) run through
Brian2 LIF neurons, with an odor->brain stimulus pathway. See `PLAN.md` for the
roadmap (odor -> brain -> decoder -> LLM).

## Running code (Windows + uv)

- Always run via the venv interpreter: `.\.venv\Scripts\python.exe <script>`.
  There is **no `pip`** in the venv; package management is `uv` only.
- Shell is PowerShell 5.1. Inline `python -c "..."` breaks on quotes; for any
  non-trivial snippet write a temp `.py` file and run it.
- Do NOT read or edit anything under `.venv/`.

## Commands

- Simulate: `.\.venv\Scripts\python.exe flynet.py --neurons 2000 --simtime 100`
  (loads the 852 MB feather, ~1 s). Writes `flynet_result.json`.
- Rebuild odor inputs: `.\.venv\Scripts\python.exe prepare_olfaction.py`
  (reads raw DoOR CSVs + annotations, writes `olfactory/*`).
- Demo odor discriminability: `.\.venv\Scripts\python.exe demo_odors.py`.
- Odor-evoked brain states (stage 3):
  `.\.venv\Scripts\python.exe run_odors.py --neurons 1500 --simtime 150
  --gain 5` — builds the AL subgraph, drives ORNs with DoOR responses,
  writes `results_odor/` (spike rasters, `states.npy`, `summary.json`).
- Inspect those results: `.\.venv\Scripts\python.exe analyze_odors.py`
  (top glomeruli per odor, population firing-rate bins).
- Build the decoder dataset (stage 4.1):
  `.\.venv\Scripts\python.exe build_odor_dataset.py --n-odors 36 --trials 20
  --drive-sigma 0.15` — repeated trials with stimulus jitter for many odors,
  writes `decoder/dataset.npz` (X_counts / X_bins / X_glom / X_glom_bins),
  `decoder/odor_names.json`, `decoder/meta.json`. Use `--workers N` to run
  trials in parallel across CPU cores (numpy device; speeds up ~2x at scale).
  Drive vectors are pre-drawn in the parent with `--seed`, so the dataset is
  bit-identical to a serial run.
  `decoder/odor_names.json`, `decoder/meta.json`. Takes ~6 min.
- Decoder baselines / ablations / open-set (stages 4.2-4.5):
  `.\.venv\Scripts\python.exe decoder_baseline.py` (closed + held-out,
  `decoder/baseline.json`), `decoder_ablation.py` (ORN/non-ORN/all,
  `decoder/ablation.json`), `decoder_open.py` (MLP closed + brain→embedding
  regression for unseen odors, `decoder/open.json`). scikit-learn is installed.
  NOTE: `decoder/dataset.npz` is currently generated at gain=40 (stronger drive;
  min odor-state corr 0.78 vs 0.92 at gain=5) for better separability.
  ORN drive time course is selectable via `--pulse tonic|bump|adapt` (timed
  TimedArray in run_odors.build_network; default tonic). Pulse drive was tried
  for stage 4.8 and did NOT improve held-out — see PLAN.md.
- Contrastive / relational metric learning (stage 4.7):
  `.\.venv\Scripts\python.exe decoder_contrast.py` — PCA features + InfoNCE
  (plain) and relational (inner-product == DoOR similarity) encoders; evaluates
  held-out retrieval against DoOR-neighbors, writes `decoder/contrast.json`.
  Plain InfoNCE hurts held-out (erases chemical structure); relational is the
  right objective. `sweep_analyze.py <sweepdir>` reports separability metrics.

## Hard constraints (do not break)

- **numpy must stay 1.26.4.** Brian2 2.8.0 is incompatible with numpy>=2
  (`np.ndarray.ptp` was removed). Never "upgrade" numpy in this venv.
- **brian2cuda/GPU does not work on native Windows** (brian2cuda issue #225).
  Do not retry; CPU (numpy) device is the working path. GPU requires WSL2/Linux,
  which is not installed.
- **cpp_standalone is NOT usable for the dataset generator.** Two reasons:
  (1) it forbids multiple `run()` per build, but each trial already does
  rest→stim→rest runs; (2) the per-neuron time-varying ORN drive would have to
  be baked into the C++ source as a huge literal. Use CPU multiprocessing
  (`--workers N`) instead — it keeps exact dataset semantics.
- `proofread_connections_783.feather` is a read-only input: columns
  `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`, transmitter
  averages. 16.8M rows, 138k neurons. Do not modify or regenerate it.

## Data quirks (easy to trip on)

- The DoOR source CSVs (`door_mappings.csv`, `door_response_matrix.csv`,
  `door_odor.csv`) are R exports: **semicolon-separated** (`sep=";"`), quotes
  around every field, NaN-heavy. `door_response_matrix.csv` is 691 odors
  (rows, InChIKey) x 78 receptors (cols); index rows align 1:1 with
  `door_odor.csv` InChIKeys.
- `flywire_neuron_annotations.tsv` is **tab-separated** (139k neurons).
  ORNs = `cell_class == "olfactory"`; their `cell_type` is `ORN_<glomerulus>`
  (e.g. `ORN_DM1`). Root ids are in the same id space as the feather and are
  all present in the connectome. Some rows have `cell_type` NaN — drop them.
- Annotations/connectome lookups must use root ids from `olfactory/orn_table.csv`
  (built once by `prepare_olfaction.py`), not re-derived from scratch.
- `flynet.log` and `flynet_result.json` are generated artifacts, not sources.
- `results_odor/` is a generated stage-3 artifact (spike rasters + states).
- `decoder/` is a generated stage-4 artifact (dataset.npz + metadata).
- sklearn `LinearSVC` **hangs** on these datasets; use `SGDClassifier(loss="hinge")`
  as the linear-SVM baseline instead. `LinearSVC` with 37 classes / 3097+ feats
  never returned (tested). Large representations (X_bins = 7×3097 feats): avoid
  broadcasting (n × n_seen × feat) distance tensors — compute pairwise dist
  per-prototype, else it OOMs (>10 GB).

## Model (flynet.py / run_odors.py) summary

LIF with exponential current synapses: tau=10 ms, Vt=-50 mV, Vr=-65 mV,
tau_syn=2 ms, dt=0.1 ms. Connection sign from `gaba_avg > glut_avg`
(inhibitory -> negative weight); magnitude `max(0.1, syn_count)`. A `--neurons`
subgraph is extracted per `--neuropil` (default EB); 1/3 of neurons get
`I_drive = 3.5 pA` (seed 42) to keep the net active.

`run_odors.py` keys: ORNs are driven by `I_inj = base + gain * DoOR_response`
with **base below spiking threshold** (default 1.0 pA). Raising base too high
(e.g. 2.2 pA) saturates ORNs at max rate and **all odors become identical**
(cos=1.0). Blank trials are silent because the model has no noise/spontaneous
activity; this is expected, not a bug. `I_drive` for non-ORN neurons is 0 in
this script (activity is purely odor-driven).