"""Build and cache the olfactory input pipeline for Talking Fly.

Produces (inside ./olfactory):
  orn_table.csv     - one row per FlyWire ORN root_id: cell_type (ORN_<glomerulus>),
                      side, and the DoOR receptor column giving its odor responses.
  gl_map.csv        - glomerulus -> receptor basis used.
  resp_matrix.npy   - (n_orns x n_odors) float32 response matrix, NaN for ORNs
                      without data. Row order == orn_table.csv row order.
  odor_names.json   - {matrix_column_index: odor name} (DoOR InChIKey -> common name).

Data sources:
  flywire_neuron_annotations.tsv   - FlyWire v783 neuron annotations (cell types).
  door_mappings.csv                - DoOR receptor <-> glomerulus mapping.
  door_response_matrix.csv         - DoOR merged odor-response matrix (691 odors).
  door_odor.csv                    - odor metadata (InChIKey -> common name).

Usage:
    python prepare_olfaction.py [--ann ...] [--outdir olfactory]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def build(ann_tsv="flywire_neuron_annotations.tsv",
          mappings_csv="door_mappings.csv",
          resp_csv="door_response_matrix.csv",
          odor_csv="door_odor.csv",
          outdir="olfactory"):
    ann = pd.read_csv(ann_tsv, sep="\t", low_memory=False)
    orn = ann[(ann["cell_class"].astype(str) == "olfactory")
              & ann["root_id"].notna()].copy()
    orn["gl"] = orn["cell_type"].astype(str).str.replace("ORN_", "", regex=False)
    orn = orn[orn["gl"].notna() & (orn["gl"] != "nan")]

    m = pd.read_csv(mappings_csv, sep=";")
    resp = pd.read_csv(resp_csv, sep=";", index_col=0)
    resp_cols = set(resp.columns)

    # receptor list per glomerulus (also relaxes "DL2d/v" -> DL2d/DL2v)
    m["gl_key"] = m["glomerulus"].astype(str)
    m = m[m["gl_key"] != "?"]

    def receptors_for(gl):
        direct = m[m["gl_key"] == gl]["receptor"].tolist()
        if direct:
            return direct
        prefix = m[m["gl_key"].str.startswith(gl + "/", na=False)]
        return prefix["receptor"].tolist()

    orn["receptors"] = orn["gl"].apply(receptors_for)
    orn["basis"] = orn["receptors"].apply(
        lambda recs: next((x for x in recs if x in resp_cols), None))

    # per-ORN response vector (NaN gap -> fills with NaN for missing data)
    n_odors = resp.shape[0]
    matrix = np.full((len(orn), n_odors), np.nan, dtype=np.float32)
    col_index = {c: i for i, c in enumerate(resp.columns)}
    for i, basis in enumerate(orn["basis"].tolist()):
        if basis is None \
                or (isinstance(basis, float) and np.isnan(basis)):
            continue
        matrix[i] = resp.iloc[:, col_index[basis]].to_numpy()

    # odor names aligned to matrix columns (rows)
    od = pd.read_csv(odor_csv, sep=";")
    ik2name = dict(
        zip(od["InChIKey"].astype(str), od["Name"].astype(str), strict=False))
    names = []
    for key in resp.index.astype(str):
        n = ik2name.get(key, key)
        names.append(n if str(n) != "nan" else key)
    with open(os.path.join(outdir, "odor_names.json"), "w") as f:
        json.dump(names, f)

    gl_map = (orn[["gl", "basis"]].drop_duplicates()
              .sort_values("gl")
              .reset_index(drop=True)
              .rename(columns={"basis": "basis_receptor"}))
    gl_map["all_receptors"] = gl_map["gl"].map(
        {gl: recs for gl, recs in zip(orn["gl"], orn["receptors"], strict=False)})

    out = orn[["root_id", "cell_type", "gl", "side", "basis"]].rename(
        columns={"basis": "basis_receptor"})
    out["has_data"] = out["basis_receptor"].notna()

    os.makedirs(outdir, exist_ok=True)
    out.to_csv(os.path.join(outdir, "orn_table.csv"), index=False)
    gl_map.to_csv(os.path.join(outdir, "gl_map.csv"), index=False)
    np.save(os.path.join(outdir, "resp_matrix.npy"), matrix)

    n_with = int(out["has_data"].sum())
    print(f"ORN neurons : {len(out)}  (with odor data: {n_with})")
    print(f"ORN types   : {orn['gl'].nunique()}")
    print(f"odors       : {n_odors}")
    print(f"files written to {outdir}/: orn_table.csv, gl_map.csv, "
          f"resp_matrix.npy, odor_names.json")
    return out, gl_map, matrix, names


def load(outdir="olfactory"):
    """Load cached olfactory inputs."""
    orn = pd.read_csv(os.path.join(outdir, "orn_table.csv"))
    matrix = np.load(os.path.join(outdir, "resp_matrix.npy"))
    with open(os.path.join(outdir, "odor_names.json")) as f:
        names = json.load(f)
    return orn, matrix, names


def find_odor(name, names, tol=0.4):
    """Fuzzy-match an odor name to a matrix index (case-insensitive)."""
    import difflib
    best, idx = None, -1
    for i, n in enumerate(names):
        score = difflib.SequenceMatcher(a=name.lower(), b=str(n).lower()).ratio()
        if score > tol and (best is None or score > best):
            best, idx = score, i
    return idx


def stimulus(odor_idx, gain_pA=30.0, base_pA=0.0, outdir="olfactory"):
    """Turn one odor into per-ORN drive currents.

    Returns a DataFrame with columns root_id, cell_type and I_inj_pA.
    ORNs without response data for the odor get I_inj = base_pA.
    """
    orn, matrix, _ = load(outdir)
    vec = matrix[:, odor_idx]
    i_inj = np.where(np.isnan(vec), base_pA, base_pA + gain_pA * vec)
    return pd.DataFrame({
        "root_id": orn["root_id"],
        "cell_type": orn["cell_type"],
        "I_inj_pA": i_inj,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann", default="flywire_neuron_annotations.tsv")
    p.add_argument("--mappings", default="door_mappings.csv")
    p.add_argument("--responses", default="door_response_matrix.csv")
    p.add_argument("--odor-meta", default="door_odor.csv")
    p.add_argument("--outdir", default="olfactory")
    args = p.parse_args()
    build(
        ann_tsv=args.ann,
        mappings_csv=args.mappings,
        resp_csv=args.responses,
        odor_csv=args.odor_meta,
        outdir=args.outdir,
    )