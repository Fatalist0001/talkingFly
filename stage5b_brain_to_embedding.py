"""Stage 5b: DIRECT brain -> text-embedding bridge (no odor classification).

Current stage 5 decodes the odor NAME first and lets LLM knowledge translate
it into an everyday smell.  This script builds the originally intended path:

    brain state -> vector in TEXT-EMBEDDING space -> nearest sensory phrases

Steps
  1. llama-server is (re)started with --embeddings --pooling mean so that
     /v1/embeddings works on the same local gemma model used for speech.
  2. A hand-written corpus of Russian sensory phrases (3 per odor + blank
     phrases) is embedded.  Odor-level targets = normalized mean of their
     phrase embeddings -> decoder/text_embeddings.npy (36 x dim), aligned
     with odor_names.json, usable by decoder_open.py as external target.
  3. Two bridges are trained from X_glom_bins and evaluated with the usual
     held-out-every-4th protocol:
       RIDGE      feature PCA -> RidgeCV -> text-PCA(32) -> inverse to full
                  space; metrics: closed phrase-retrieval acc@1/@3,
                  open-set cos(pred, TRUE held-out text emb) vs ceiling.
       RELATIONAL metric learning (decoder_contrast.train_relational) with
                  inner product == TEXT similarity; held-out mean_rank under
                  text-similarity ground truth (DoOR ranks reported too).
  4. The deployment model (feature PCA + text PCA + ridge on ALL odor trials,
     phrase embeddings) is saved to decoder/bridge_emb.npz for
     stage5_fly_speaks.py --bridge emb.

Usage:
  .\\.venv\\Scripts\\python.exe stage5b_brain_to_embedding.py
  .\\.venv\\Scripts\\python.exe stage5b_brain_to_embedding.py --offline
      (reuse cached phrase embeddings, no server needed)
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from stage5_fly_speaks import (DATA_DIR, http_json, server_ok, stop_server,
                               load_dataset)

# ---- sensory corpus ----------------------------------------------------------
# 3 short "what it smells like" sentences per dataset odor (no chemistry),
# plus blank phrases.  Keys must match decoder/odor_names.json exactly.
PHRASES = {
    "water": ["почти не пахнет, просто влажность", "капля воды, лёгкая свежесть",
              "чистая влага без запаха"],
    "gamma-hexalactone": ["пахнет кокосом и спелым персиком",
                          "сладкий сливочно-фруктовый аромат",
                          "десерт с кокосом и персиком"],
    "3-methylthio-1-propanol": ["пахнет варёным картофелем и супом",
                                "овощной отвар, что-то сварилось",
                                "тёплый запах кухни: картошка и бульон"],
    "alpha-terpineol": ["пахнет сиренью с цитрусом и хвоей",
                        "цветущий сад и хвойная свежесть",
                        "лимонная корка среди веток сирени"],
    "(-)-menthone": ["пахнет мятой", "холодный мятный запах",
                     "свежесть размятых мятных листьев"],
    "beta-ionone": ["пахнет фиалками", "нежный цветочный запах фиалок",
                    "весенние фиалки в саду"],
    "methyl salicylate": ["пахнет мазью звёздочка и мятной жвачкой",
                          "аптечная мазь с ментолом",
                          "резкая мятно-лекарственная свежесть"],
    "methyl benzoate": ["пахнет фейхоа", "фруктово-цветочный запах фейхоа",
                        "сладкий цветок с фруктовым соком"],
    "benzaldehyde": ["пахнет горьким миндалём и марципаном",
                     "миндальное печенье с горчинкой",
                     "сладкий марципан и капля аптеки"],
    "2-propylphenol": ["пахнет аптечным антисептиком и дёгтем",
                       "резкий дегтярный запах из аптеки",
                       "лекарство и смола"],
    "methyl jasmonate": ["пахнет жасмином", "густой вечерний аромат жасмина",
                         "белая жасминовая ветка в цвету"],
    "2-ethylphenol": ["пахнет дымом и кожей",
                      "костровый дым и выделанная кожа",
                      "копчёная кожа у огня"],
    "1-pentanol": ["пахнет скошенной травой и спиртом",
                   "зелёная трава после покоса с резинкой спирта",
                   "свежескошенный газон"],
    "1-hexanol": ["пахнет травой и зелёным яблоком",
                  "свежая зелень с яблочной кислинкой",
                  "садовая трава и неспелые яблоки"],
    "1-octanol": ["пахнет воском и апельсиновой коркой",
                  "цитрусовая корка со свечным воском",
                  "жирный блеск воска и апельсина"],
    "2-hexanol": ["пахнет спиртом с травяной нотой",
                  "резкий спирт и зелень",
                  "брага среди травы"],
    "butyl acetate": ["пахнет бананом и лаком для ногтей",
                      "сладкий банан в химическом лаке",
                      "фруктовый лак, как в маникюрном салоне"],
    "hexyl butyrate": ["пахнет яблоком и грушей",
                       "сочный микс яблока с грушей",
                       "фруктовый сад наливается соком"],
    "ethyl lactate": ["пахнет кислым молоком",
                      "ацидофилин и простокваша",
                      "кисломолочный запах холодильника"],
    "octyl acetate": ["пахнет апельсином", "спелый цитрус, апельсиновая корка",
                      "свежий апельсиновый сок"],
    "ethyl 2-methylbutanoate": ["пахнет яблочной конфетой и ананасом",
                                "сладкая фруктовая карамель с ананасом",
                                "леденец со вкусом яблока"],
    "heptyl acetate": ["пахнет спелой грушей",
                       "мягкий сладкий грушевый аромат",
                       "груша переспела и пахнет на всю комнату"],
    "beta-butyrolactone": ["пахнет сладкой карамелью",
                           "жжёный сахар и тянущаяся карамель",
                           "конфетка пахнет жжёным сахаром"],
    "2-methylisoborneol": ["пахнет тиной и илистой рекой",
                           "сырая речная тина",
                           "болото и старая вода"],
    "2-methoxy-4-vinyl phenol": ["пахнет гвоздикой и копчёностями",
                                 "пряная гвоздика над коптильней",
                                 "жжёная пряность с дымком"],
    "(2S)-heptan-2-ol": ["пахнет спиртом с грибной нотой",
                         "резкий алкоголь и подвал с грибами",
                         "самогон да лесные грибы"],
    "cis-2-hexenyl crotonate": ["пахнет зелёным яблоком и травой",
                                "терпкая зелень неспелых плодов",
                                "луговая трава и падалица"],
    "beta-himachalene": ["пахнет кедровым деревом",
                         "тёплый запах столярки: кедровые опилки",
                         "распилили смолистое дерево"],
    "beta-elemene": ["пахнет деревом и травяным маслом",
                     "древесная стружка с маслом",
                     "аптечное древесно-травяное масло"],
    "4-methyl-2-nitrophenol": ["пахнет лекарством с дымком",
                               "горькое снадобье у костра",
                               "аптечная горечь и дымок"],
    "(R)-(-)-1-octen-3-ol": ["пахнет лесными грибами",
                             "грибница во влажном лесу",
                             "сырые грибы после дождя"],
    "phenylacetaldehyde dimethyl acetal": ["пахнет зелёным гиацинтом",
                                           "свежий весенний гиацинт",
                                           "зелёные ростки гиацинта в горшке"],
    "m-cymene": ["пахнет цитрусом и скипидаром",
                 "лимонная краска на скипидаре",
                 "технический цитрус, малярка"],
    "isoamyl isovalerate": ["пахнет яблоком", "спелая антоновка",
                            "яблоневый сад в пору сбора"],
    "4-methylthiazole": ["пахнет жареным мясом", "мясо на сковородке",
                         "подгоревшая мясная поджарка"],
    "7-oxabicyclo[2.2.1]-heptane": ["пахнет техническим растворителем",
                                    "едкая мастерская химчистки",
                                    "резкий промышленный растворитель"],
}
BLANK_PHRASES = [
    "воздух чистый, ничем не пахнет", "никакого запаха, тишина",
    "антенны не чуют ничего", "пустой воздух без запаха",
    "ни еды, ни опасности — ничего не пахнет",
    "в воздухе только свежесть, без запаха",
]

TEXT_PCA_DIM = 32
FEAT_PCA_DIM = 128
REL_EMB_DIM = 64


# ---- server ------------------------------------------------------------------
def embeddings_ok(base_url):
    try:
        r = http_json(base_url + "/v1/embeddings", {"input": ["тест"]},
                      timeout=120)
        return len(r["data"][0]["embedding"]) > 0
    except Exception:
        return False


def ensure_embed_server(base_url):
    """Server up AND able to embed; restart without --embeddings if needed."""
    if server_ok(base_url):
        if embeddings_ok(base_url):
            print("[llm] server already up with embeddings")
            return False
        print("[llm] running server has no embeddings endpoint; restarting")
        stop_server(base_url)
        time.sleep(2)
    import subprocess
    import sys
    LLAMA_SERVER = r"D:\llama.cpp\build\bin\Release\llama-server.exe"
    LLAMA_MODEL = (r"D:\Lm studio\.lmstudio\models\lmstudio-community"
                   r"\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf")
    port = base_url.rsplit(":", 1)[-1]
    print(f"[llm] starting llama-server (embeddings) on port {port} ...")
    flags = 0x00000008 | 0x00000200
    subprocess.Popen(
        [LLAMA_SERVER, "-m", LLAMA_MODEL, "--host", "127.0.0.1",
         "--port", str(port), "-c", "4096", "--embeddings",
         "--pooling", "mean"],
        creationflags=flags,
        stdout=open(os.path.join(os.environ.get("TEMP", "."), "fly_llama.log"),
                    "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 300:
        time.sleep(5)
        if server_ok(base_url) and embeddings_ok(base_url):
            print(f"[llm] ready in {time.time() - t0:.0f}s")
            return True
    sys.exit("llama-server with embeddings did not become healthy in 300s")


def embed_texts(base_url, texts, batch=16):
    chunks = []
    for s in range(0, len(texts), batch):
        part = texts[s:s + batch]
        r = http_json(base_url + "/v1/embeddings", {"input": part},
                      timeout=600)
        arr = sorted(r["data"], key=lambda d: d["index"])
        chunks.append(np.array([d["embedding"] for d in arr], dtype=np.float64))
        print(f"  embedded {min(s + batch, len(texts))}/{len(texts)}",
              flush=True)
    out = np.vstack(chunks)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(nrm, 1e-9)


# ---- corpus ------------------------------------------------------------------
def build_corpus(odor_names):
    texts, owners = [], []
    for i, n in enumerate(odor_names):
        for p in PHRASES[n]:
            texts.append(p)
            owners.append(i)
    for p in BLANK_PHRASES:
        texts.append(p)
        owners.append(-1)
    return texts, np.array(owners)


def odor_targets(phrase_embs, owners, n_odors):
    T = np.zeros((n_odors, phrase_embs.shape[1]))
    for i in range(n_odors):
        rows = phrase_embs[owners == i]
        T[i] = rows.mean(0)
    # mean-center: raw mean-pooled embeddings sit in a narrow cone
    # (unrelated pairs cos~0.89), centering expands the discriminative part
    T = T - T.mean(0, keepdims=True)
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    return T / np.maximum(nrm, 1e-9)


def center_unit(M):
    M = M - M.mean(0, keepdims=True)
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(nrm, 1e-9)


def corr2(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def corr_matrix(M):
    """Pearson correlation between columns of M (n_feat, n_items)."""
    Mc = M - M.mean(axis=0, keepdims=True)
    Mn = Mc / np.maximum(np.linalg.norm(Mc, axis=0, keepdims=True), 1e-9)
    return Mn.T @ Mn


def rank_map(sim36):
    """rank_map[i][j] = rank of j among all odors by similarity to i."""
    order = np.argsort(-sim36, axis=1)
    r = np.empty_like(order)
    for i in range(len(order)):
        r[i, order[i]] = np.arange(len(order))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="reuse cached embeddings from bridge_emb.npz")
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    d, names, meta = load_dataset()
    n_odor = meta["n_odors"]
    odor_names = names[:n_odor]
    y = d["y"]
    X = d["X_glom_bins"].reshape(len(y), -1).astype(np.float64)
    odor_mask = y < n_odor
    totals = d["X_glom"].sum(1)
    blank_max = float(totals[y == y.max()].max())
    odor_min = float(totals[odor_mask].min())
    blank_thr = 0.5 * min(blank_max + 1e-6, odor_min) \
        if odor_min > blank_max else blank_max + 1e-6

    bridge_path = os.path.join(DATA_DIR, "bridge_emb.npz")
    base_url = f"http://127.0.0.1:8765"

    # ---- 1. embed the corpus -------------------------------------------------
    raw_cache = os.path.join(DATA_DIR, "phrase_embs_raw.npy")
    if args.offline and os.path.exists(raw_cache):
        with open(os.path.join(DATA_DIR, "phrase_corpus.json"),
                  encoding="utf-8") as f:
            c = json.load(f)
        texts, owners = c["texts"], np.array(c["owners"])
        phrase_embs = np.load(raw_cache)
        print(f"[corpus] reused cached embeddings {phrase_embs.shape}")
        spawned = False
    else:
        spawned = ensure_embed_server(base_url)
        texts, owners = build_corpus(odor_names)
        print(f"[corpus] embedding {len(texts)} phrases ...")
        phrase_embs = embed_texts(base_url, texts)
        np.save(raw_cache, phrase_embs)
    owners = owners.astype(int)
    # All-but-the-Top: drop the single dominant style direction
    mu = phrase_embs.mean(0, keepdims=True)
    pc1 = PCA(n_components=1).fit(phrase_embs - mu).components_
    phrase_embs = center_unit(
        (phrase_embs - mu) - ((phrase_embs - mu) @ pc1.T) @ pc1)

    # ---- 2. odor-level text targets -----------------------------------------
    T = odor_targets(phrase_embs, owners, n_odor)
    np.save(os.path.join(DATA_DIR, "text_embeddings.npy"), T)
    with open(os.path.join(DATA_DIR, "phrase_corpus.json"), "w",
              encoding="utf-8") as f:
        json.dump({"texts": texts, "owners": owners.tolist()}, f,
                  ensure_ascii=False, indent=1)
    print(f"[target] text_embeddings.npy {T.shape} saved")

    # ---- sanity: does text space agree with chemistry? -----------------------
    import prepare_olfaction as op
    _, door, all_names = op.load()
    cols = [all_names.index(n) for n in odor_names]
    door_sim = corr_matrix(np.nan_to_num(door[:, cols],
                                         nan=0.0).astype(np.float64))
    text_sim = T @ T.T                                   # cosine, unit rows
    rho = spearmanr(door_sim[np.triu_indices(n_odor, 1)],
                    text_sim[np.triu_indices(n_odor, 1)]).statistic
    print(f"[sanity] Spearman(text-sim, DoOR-sim) over all pairs: {rho:.3f}")

    # ---- splits --------------------------------------------------------------
    held = list(range(0, n_odor, 4))                     # 9 unseen odors
    seen = [c for c in range(n_odor) if c not in held]
    tr_mask = np.isin(y, seen)
    te_mask = np.isin(y, held)
    print(f"held-out ({len(held)}): {[odor_names[c] for c in held]}")

    door_rank = rank_map(door_sim)
    text_rank = rank_map(text_sim)
    Sc = np.array([[corr2(T[i], T[j]) for j in range(n_odor)]
                   for i in range(n_odor)])
    offdiag = ~np.eye(n_odor, dtype=bool)
    chance_cos = float(Sc[offdiag].mean())

    report = {"spearman_text_vs_door": float(rho),
              "chance_cos_unrelated": chance_cos}

    # ---- 3a. ridge bridge ----------------------------------------------------
    print("\n=== ridge bridge ===")
    fpca = PCA(n_components=FEAT_PCA_DIM, whiten=True)
    Ztr = fpca.fit_transform(X[tr_mask])
    Z = fpca.transform(X)
    tpca = PCA(n_components=TEXT_PCA_DIM)
    Etr = tpca.fit_transform(T)                          # unsupervised basis
    reg = RidgeCV(alphas=np.logspace(-1, 4, 12))
    reg.fit(Ztr, Etr[y[tr_mask]])
    print(f"  best alpha={reg.alpha_:.2g}, "
          f"text-PCA explained var "
          f"{tpca.explained_variance_ratio_.sum():.2f}")

    def to_full(E_low):
        E = tpca.inverse_transform(E_low)
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.maximum(nrm, 1e-9)

    pe = phrase_embs                                     # unit rows
    po = owners

    def retrieval(mask):
        pred = to_full(reg.predict(Z[mask]))
        sims = pred @ pe.T
        top = np.argsort(-sims, axis=1)[:, :3]
        true_lbl = y[mask]
        a1 = float(np.mean([po[top[i, 0]] == true_lbl[i]
                            for i in range(len(top))]))
        a3 = float(np.mean([true_lbl[i] in po[t]
                            for i, t in enumerate(top)]))
        return a1, a3, sims

    a1, a3, _ = retrieval(tr_mask)
    print(f"  closed phrase-retrieval: acc@1={a1:.3f} acc@3={a3:.3f} "
          f"(chance@1={1 / n_odor:.3f})")

    pred_held = to_full(reg.predict(Z[te_mask]))
    cos_true = np.array([corr2(pred_held[i], T[y[te_mask][i]])
                         for i in range(len(pred_held))])
    ceil = [max(corr2(T[c], T[s]) for s in seen) for c in held]
    sims_h = pred_held @ pe.T
    top_h = np.argmax(sims_h, axis=1)
    door_nb_ok = []
    lbl_by_trial = y[te_mask]
    for i, t in enumerate(top_h):
        got = po[t]
        if got < 0:
            door_nb_ok.append(False)
            continue
        door_nb_ok.append(bool(door_rank[lbl_by_trial[i], got] < 3))
    res_ridge = {
        "closed_acc@1": a1, "closed_acc@3": a3,
        "open_cos_to_true_mean": float(cos_true.mean()),
        "open_cos_to_true_std": float(cos_true.std()),
        "ceiling_true_vs_nearest_seen": float(np.mean(ceil)),
        "chance_cos": chance_cos,
        "open_top_phrase_is_doortop3_nb": float(np.mean(door_nb_ok)),
    }
    report["ridge"] = res_ridge
    print("  open-set:", json.dumps(res_ridge, ensure_ascii=False))

    print("  --- held-out examples (mean prediction -> top phrases) ---")
    for c in held:
        rows = np.where(y == c)[0]
        e = to_full(reg.predict(Z[rows])).mean(0)
        e /= max(np.linalg.norm(e), 1e-9)
        order = np.argsort(-(pe @ e))
        picked, got = [], set()
        for t in order:
            if po[t] >= 0 and po[t] not in got:
                picked.append(texts[t])
                got.add(po[t])
            if len(picked) == 2:
                break
        print(f"    {odor_names[c]:38s} -> «{picked[0]}» | «{picked[1]}»")

    # ---- 3b. relational bridge ----------------------------------------------
    print("\n=== relational bridge (inner product == text similarity) ===")
    from decoder_contrast import pca_feats, train_relational, embed
    Zc, _ = pca_feats(X, tr_mask, k=FEAT_PCA_DIM)
    W_rel = train_relational(Zc, y, np.where(tr_mask)[0],
                             list(range(n_odor)), T.T, E=REL_EMB_DIM,
                             lr=0.3, epochs=400, seed=0)
    Ez = embed(Zc, W_rel)
    cents = {l: Ez[y == l].mean(0) for l in seen}
    cents_n = {l: c / np.linalg.norm(c) for l, c in cents.items()}
    closed_hits, rel_ranks_text, rel_ranks_door = [], [], []
    for ht in np.where(te_mask)[0]:
        q = Ez[ht] / np.linalg.norm(Ez[ht])
        best = max(cents_n, key=lambda l: float(q @ cents_n[l]))
        rel_ranks_text.append(int(text_rank[y[ht], best]))
        rel_ranks_door.append(int(door_rank[y[ht], best]))
    for l in seen:
        for st in np.where(y == l)[0]:
            q = Ez[st]
            best = max(cents, key=lambda c: float(q @ cents[c]))
            closed_hits.append(best == l)
    res_rel = {
        "closed_1nn_acc": float(np.mean(closed_hits)),
        "heldout_text_rank_mean": float(np.mean(rel_ranks_text)),
        "heldout_text_rank_hit@3": float(np.mean(
            np.array(rel_ranks_text) < 3)),
        "chance_text_rank": float((n_odor - 1) / 2),
        "heldout_door_rank_mean": float(np.mean(rel_ranks_door)),
    }
    report["relational"] = res_rel
    print("  ", json.dumps(res_rel))

    # ---- 3c. hybrid bridge: brain -> predicted DoOR profile -> text mix ------
    # pure text targets do not carry receptor geometry (see Spearman above),
    # so for UNSEEN odors we predict the chemical profile first and build the
    # meaning vector as a soft mixture of SEEN odors' text centroids.
    print("\n=== hybrid bridge (brain->DoOR->text mixture) ===")
    from decoder_contrast import train_reg78
    Mdoor = np.nan_to_num(door[:, cols], nan=0.0).astype(np.float64)
    # similarity basis: CENTERED unit odor profiles (raw non-negative
    # profiles all correlate ~0.55+ and the landscape is flat)
    Msim = Mdoor - Mdoor.mean(0, keepdims=True)
    Msim /= np.maximum(np.linalg.norm(Msim, axis=0, keepdims=True), 1e-9)
    W78 = train_reg78(Zc, y, np.where(tr_mask)[0], list(range(n_odor)),
                      Mdoor, E=door.shape[0], lr=0.1, epochs=400, seed=0)

    def door_to_text(zrows, temp=0.05):
        zc = zrows - zrows.mean(1, keepdims=True)
        sims = zc @ Msim                              # (n, 36)
        w = np.exp((sims - sims.max(1, keepdims=True)) / temp)
        w /= w.sum(1, keepdims=True)
        V = w @ T                                    # mixture of centroids
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.maximum(nrm, 1e-9)

    Z78 = embed(Zc, W78)
    Vh = door_to_text(Z78)
    odor_rows = np.where(np.isin(y, seen))[0]
    top1 = np.argmax(Vh[odor_rows] @ pe.T, axis=1)
    h_a1 = float(np.mean([owners[top1[i]] == y[odor_rows][i]
                          for i in range(len(top1))]))
    ranks_h = []
    for c in held:
        rows = np.where(y == c)[0]
        v = Vh[rows].mean(0)
        v /= max(np.linalg.norm(v), 1e-9)
        sc = v @ T.T
        ranks_h.append(int(np.where(np.argsort(-sc) == c)[0][0]))
    res_hyb = {"closed_acc@1": h_a1,
               "heldout_target_rank_mean": float(np.mean(ranks_h))}
    report["hybrid"] = res_hyb
    print("  ", json.dumps(res_hyb))
    print("  --- held-out examples ---")
    for c in held:
        rows = np.where(y == c)[0]
        v = Vh[rows].mean(0)
        v /= max(np.linalg.norm(v), 1e-9)
        order = np.argsort(-(pe @ v))
        picked, got = [], set()
        for t in order:
            if po[t] >= 0 and po[t] not in got:
                picked.append(texts[t])
                got.add(po[t])
            if len(picked) == 2:
                break
        print(f"    {odor_names[c]:38s} -> «{picked[0]}» | «{picked[1]}»")

    # ---- 4. deployment model (refit on ALL odor trials) ----------------------
    fpca_f = PCA(n_components=FEAT_PCA_DIM, whiten=True)
    Zf = fpca_f.fit_transform(X[odor_mask])
    tpca_f = PCA(n_components=TEXT_PCA_DIM)
    Ef = tpca_f.fit_transform(T)
    reg_f = RidgeCV(alphas=np.logspace(-1, 4, 12))
    reg_f.fit(Zf, Ef[y[odor_mask]])
    W78_f = train_reg78(Zf, y, np.where(odor_mask)[0], list(range(n_odor)),
                        Mdoor, E=door.shape[0], lr=0.1, epochs=400, seed=0)
    np.savez(bridge_path,
             fpca_mean=fpca_f.mean_, fpca_comps=fpca_f.components_,
             fpca_scale=np.sqrt(fpca_f.explained_variance_),
             tpca_mean=tpca_f.mean_, tpca_comps=tpca_f.components_,
             ridge_coef=reg_f.coef_, ridge_intercept=reg_f.intercept_,
             reg78_coef=W78_f, door_cols=Msim,
             text_targets=T.astype(np.float32), hybrid_temp=np.float64(0.05),
             phrase_embs=pe, phrase_texts=np.array(texts),
             phrase_owners=owners.astype(np.int64),
             blank_thr=blank_thr)
    with open(os.path.join(DATA_DIR, "bridge_emb.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {bridge_path}, {DATA_DIR}/text_embeddings.npy, "
          f"{DATA_DIR}/bridge_emb.json  ({time.perf_counter() - t0:.1f}s)")

    if spawned and not args.keep_server:
        stop_server(base_url)


if __name__ == "__main__":
    main()
