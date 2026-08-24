"""Stage 5: odor -> fly brain -> decoder -> context -> LLM phrase.

The full TalkingFly chain, live:
  1. one fresh trial is simulated in the canonical AL network (gain=40,
     per-edge delays N(1.5,0.5) ms - same recipe as decoder/dataset.npz);
  2. the brain state is decoded by a closed-set logistic regression trained
     on the canonical dataset (glomerulus-binned features), with a trivial
     silence detector for blanks;
  3. chemical context (nearest DoOR neighbors of the decoded odor,
     stimulus intensity) is assembled;
  4. a local LLM (llama.cpp server, default gemma E2B) turns the decoded
     smell into one short phrase spoken by the fly.

Usage:
  python stage5_fly_speaks.py                      # random odor, fresh sim
  python stage5_fly_speaks.py --odor benzaldehyde  # specific odor
  python stage5_fly_speaks.py --list               # show available odors
  python stage5_fly_speaks.py --blank              # empty air
  python stage5_fly_speaks.py --dataset-trial      # reuse dataset row (fast)
  python stage5_fly_speaks.py --no-llm             # decode only

The script starts llama-server itself if it is not already running.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import types
import urllib.request

import numpy as np

if hasattr(sys.stdout, "reconfigure"):          # cyrillic-safe printing
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = "decoder"
LLAMA_SERVER = r"D:\llama.cpp\build\bin\Release\llama-server.exe"
LLAMA_MODEL = (r"D:\Lm studio\.lmstudio\models\lmstudio-community"
               r"\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf")
PORT = 8765


# ---- data ------------------------------------------------------------------
def load_dataset(data_dir=DATA_DIR):
    d = np.load(os.path.join(data_dir, "dataset.npz"))
    with open(os.path.join(data_dir, "odor_names.json")) as f:
        names = json.load(f)
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    return d, names, meta


def train_decoder(d):
    """Closed-set logreg on flattened glom-binned features + blank stats."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xg = d["X_glom_bins"]                       # (T, B, D)
    y = d["y"]
    T, B, D = Xg.shape
    X = Xg.reshape(T, B * D).astype(np.float64)

    totals = d["X_glom"].sum(1)                 # per-trial ORN spike total
    blank_max = float(totals[y == y.max()].max())
    odor_min = float(totals[y < y.max()].min())
    blank_thr = 0.5 * min(blank_max + 1e-6, odor_min) \
        if odor_min > blank_max else blank_max + 1e-6

    mask = y < y.max()                          # odors only
    sc = StandardScaler().fit(X[mask])
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(sc.transform(X[mask]), y[mask])
    return dict(scaler=sc, clf=clf, blank_thr=blank_thr,
                blank_max=blank_max, odor_min=odor_min)


def door_neighbors(matrix, ds_door_idx, oi, odor_names, decoded_name, k=3):
    """Top-k dataset odors closest to `oi` by DoOR receptor-profile corr."""
    clean = np.nan_to_num(matrix, nan=0.0)
    v = clean[:, oi]
    sims = []
    for ki, j in enumerate(ds_door_idx):
        if j < 0 or odor_names[ki] == decoded_name:
            continue
        w = clean[:, j]
        if v.std() == 0 or w.std() == 0:
            continue
        sims.append((odor_names[ki], float(np.corrcoef(v, w)[0, 1])))
    sims.sort(key=lambda t: -t[1])
    return sims[:k]


# ---- live simulation --------------------------------------------------------
def simulate_state(odor_name, seed):
    """One fresh trial through the canonical network; returns glom_bins."""
    import prepare_olfaction as op
    from build_odor_dataset import simulate_trial
    from run_odors import build_subgraph_real, weight_transform

    d, _, meta = load_dataset()
    ref_ids = d["root_ids"]

    orn, matrix, all_names = op.load()
    orn_ids = orn["root_id"].astype(np.int64).to_numpy()
    orn_ct = dict(zip(orn_ids, orn["cell_type"], strict=False))
    oi = op.find_odor(odor_name, all_names)
    if oi < 0:
        sys.exit(f"odor '{odor_name}' not found in DoOR")

    chosen, i_arr, j_arr, w_syn, is_orn, delays = build_subgraph_real(
        "proofread_connections_783.feather", 1500, orn_ids,
        syn_mode="sign", delay_mean_ms=meta["delay_mean_ms"],
        delay_std_ms=meta["delay_std_ms"], seed=0)
    w_syn = weight_transform(w_syn, i_arr, j_arr, len(chosen),
                             meta.get("weight_transform", "baseline"))

    # align neuron indexing to the saved dataset order
    pos = {int(r): k for k, r in enumerate(ref_ids)}
    perm = np.array([pos[int(c)] for c in chosen])
    inv = np.argsort(perm)                      # ref_idx -> chosen_idx
    i_arr = perm[i_arr]
    j_arr = perm[j_arr]
    is_orn = is_orn[inv]

    N = len(ref_ids)
    glom_types = list(meta["glom_names"])
    type_id = {t: i for i, t in enumerate(glom_types)}
    group_ids = np.array([type_id[orn_ct.get(int(n), "other")]
                          for n in ref_ids])
    onehot = np.zeros((N, len(glom_types)), dtype=np.float32)
    onehot[np.arange(N), group_ids] = 1.0

    resp = matrix[:, oi]
    v = np.where(np.isnan(resp), 0.0, resp)
    rng = np.random.default_rng(seed)
    vj = v * (1.0 + meta["drive_sigma"] * rng.standard_normal(int(is_orn.sum())))
    amp = np.zeros(N)
    amp[is_orn] = np.maximum(meta["gain_pA"] * vj, 0.0)

    args = types.SimpleNamespace(
        base=meta["base_pA"], pulse=meta["pulse"],
        stim_start=meta["stim_start_ms"], stim_dur=meta["stim_dur_ms"],
        simtime=meta["simtime_ms"], nbins=meta["nbins"],
        traj_bins=meta.get("traj_bins", 0))
    counts, bins, _ = simulate_trial(amp, args, N, is_orn, i_arr, j_arr,
                                     w_syn, delays)
    return bins @ onehot, oi, all_names, matrix


# ---- LLM --------------------------------------------------------------------
def http_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url, json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def server_ok(base_url):
    try:                                    # /health is GET-only
        with urllib.request.urlopen(base_url + "/health", timeout=3) as r:
            return json.loads(r.read().decode("utf-8")).get("status") == "ok"
    except Exception:
        return False


def ensure_server(base_url):
    """Make sure the server is up. Returns True if THIS call spawned it."""
    if server_ok(base_url):
        print(f"[llm] server already up at {base_url}")
        return False
    print(f"[llm] starting llama-server on port {PORT} ...")
    flags = 0x00000008 | 0x00000200                  # DETACHED | NO_WINDOW
    subprocess.Popen(
        [LLAMA_SERVER, "-m", LLAMA_MODEL, "--host", "127.0.0.1",
         "--port", str(PORT), "-c", "4096"],
        creationflags=flags,
        stdout=open(os.path.join(os.environ.get("TEMP", "."), "fly_llama.log"),
                    "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        time.sleep(5)
        if server_ok(base_url):
            print(f"[llm] ready in {time.time() - t0:.0f}s")
            return True
    sys.exit("llama-server did not become healthy in 240s")


def stop_server(base_url):
    """Kill whatever llama-server listens on our port (best effort)."""
    port = base_url.rsplit(":", 1)[-1]
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True, timeout=30).stdout
        pids = {ln.split()[-1] for ln in out.splitlines()
                if f":{port}" in ln and "LISTENING" in ln}
        killed = []
        for pid in pids:
            r = subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                killed.append(pid)
        if killed:
            print(f"[llm] stopped llama-server (pid {' '.join(killed)})")
            return True
    except Exception as e:
        print(f"[llm] shutdown failed: {e}")
    return False


def ask_llm(base_url, system, user, temperature=0.9, max_tokens=120):
    out = http_json(base_url + "/v1/chat/completions", {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def smell_description(base_url, name):
    """Step 1 of the semantic bridge: chemical name -> everyday smell."""
    desc = ask_llm(
        base_url,
        "Ты — справочник по запахам. Отвечай по-русски ОДНЫМ коротким "
        "словосочетанием (2-5 слов), как вещество пахнет в обычной жизни. "
        "Без вступлений и пояснений.",
        f"Чем пахнет {name}?", temperature=0.2, max_tokens=40)
    return desc or "неизвестным веществом"


def fly_phrase(base_url, facts):
    system = (
        "Ты — дрозофила, маленькая мушка. Ты не ассистент и не учёный: ты "
        "живое насекомое, тебя волнуют еда, опасность и другие мухи. Тебе "
        "сообщают, что почувствовали твои антенны. Ответь РОВНО одной "
        "короткой фразой (до 15 слов) от первого лица по-русски, живо и "
        "по-мушиному. Не упоминай химию, рецепторы, проценты и то, что ты "
        "LLM. Если сказано, что запаха нет — НЕ выдумывай никакой запах. "
        "Никаких пояснений — только сама фраза.")
    return ask_llm(base_url, system, "Что чувствуют антенны: " + facts)


# ---- main -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--odor", default=None,
                   help="odor name (substring match); default random")
    p.add_argument("--blank", action="store_true", help="simulate empty air")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-trial", action="store_true",
                   help="take a stored dataset row instead of simulating")
    p.add_argument("--list", action="store_true",
                   help="list dataset odors and exit")
    p.add_argument("--no-llm", action="store_true", help="decode only")
    p.add_argument("--shutdown", action="store_true",
                   help="stop the llama-server and exit")
    p.add_argument("--keep-server", action="store_true",
                   help="do not stop the server after the run "
                        "(default: a server started by this run is stopped)")
    p.add_argument("--server-url", default=f"http://127.0.0.1:{PORT}")
    args = p.parse_args()

    if args.shutdown:
        if stop_server(args.server_url):
            return
        print("[llm] no server running on this port")
        return

    d, names, meta = load_dataset()
    n_odors = meta["n_odors"]
    odor_names = names[:n_odors]
    if args.list:
        print("\n".join(odor_names))
        return
    rng = np.random.default_rng(args.seed)

    # ---- obtain a brain state -------------------------------------------------
    if args.dataset_trial:
        if args.blank:
            rows = np.where(d["y"] == n_odors)[0]
        elif args.odor:
            cand = [i for i, n in enumerate(odor_names)
                    if args.odor.lower() in n.lower()]
            if not cand:
                sys.exit(f"no dataset odor matches '{args.odor}'")
            rows = np.where(np.isin(d["y"], cand))[0]
        else:
            rows = np.where(d["y"] < n_odors)[0]
        t_idx = int(rng.choice(rows))
        state = d["X_glom_bins"][t_idx].astype(np.float64)
        truth = odor_names[d["y"][t_idx]] if d["y"][t_idx] < n_odors \
            else "__blank__"
        source = f"dataset row {t_idx}"
        dec_oi = None
    else:
        if args.blank:
            state = np.zeros((meta["nbins"], len(meta["glom_names"])))
            truth, source, dec_oi = "__blank__", "fresh simulation", None
        else:
            print("[brain] loading connectome and building network ...")
            odor = args.odor
            if odor is None:
                import prepare_olfaction as op
                _, matrix, all_names = op.load()
                pool = [n for n in odor_names if n != "__blank__"
                        and op.find_odor(n, all_names) >= 0]
                odor = pool[int(rng.integers(len(pool)))]
            state, dec_oi, all_names, matrix = simulate_state(odor, 
                                                              args.seed or 0)
            truth, source = all_names[dec_oi], "fresh simulation"

    # ---- decode ---------------------------------------------------------------
    model = train_decoder(d)
    total = float(state.sum())
    if total <= model["blank_thr"]:
        verdict, top, conf = "__blank__", [], 0.0
    else:
        X = state.reshape(1, -1)
        pr = model["clf"].predict_proba(model["scaler"].transform(X))[0]
        order = np.argsort(pr)[::-1][:3]
        top = [(odor_names[i], float(pr[i])) for i in order]
        verdict, conf = top[0]

    print(f"\n=== pipeline trace ({source}) ===")
    print(f"truth: {truth}   ORN spikes in window: {total:.0f}")
    if top:
        print("decoder:", ", ".join(f"{n} {c:.0%}" for n, c in top))

    if args.no_llm:
        return

    # ---- context + phrase -----------------------------------------------------
    spawned = ensure_server(args.server_url)
    med = float(np.median(d["X_glom"].sum(1)[d["y"] < n_odors]))
    if verdict == "__blank__":
        facts = ("сейчас НИЧЕГО не пахнет, воздух чистый. Скажи одной "
                 "фразой, что не чуешь ничего (можно пожаловаться на "
                 "скуку или полетать вхолостую).")
        intro = "[пустой воздух]"
    else:
        intensity = ("едва уловимый" if total < 0.33 * med
                     else "отчётливый" if total < 2.0 * med
                     else "очень резкий")
        neigh = ""
        if dec_oi is not None:
            import prepare_olfaction as op
            _, matrix, all_names = op.load()
            ds_door_idx = [op.find_odor(n, all_names) for n in odor_names]
            nn = door_neighbors(matrix, ds_door_idx, dec_oi, odor_names,
                                verdict)
            neigh = ", ".join(n for n, _ in nn[:3])
            neigh = f" Похожие запахи: {neigh}."
        smell = smell_description(args.server_url, verdict)
        print(f"[смысл: пахнет как {smell}]")
        facts = (f"запах {intensity}, похож на {smell}"
                 f" (вещество {verdict}).{neigh}")
        intro = f"[декодировано: {verdict} {conf:.0%}]"
    print(f"\n{intro}\n[муха думает...]")
    phrase = fly_phrase(args.server_url, facts)
    print(f"\nМуха: «{phrase}»")
    if spawned and not args.keep_server:
        stop_server(args.server_url)


if __name__ == "__main__":
    main()
