#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TabNet on synthetic CBC (no embedding files):
- Unsupervised TabNetPretrainer on train
- Extract row embeddings as the sum of decision-step outputs (encoder rep)
- Same metrics + AUCs; optional PCA→4 variant
"""

import os, re, zipfile, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import trustworthiness as skl_trustworthiness
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- config ----------
DATA_ARCHIVE = "synthetic_cbc.zip"
OUTPUT_DIR   = "TabNetOutputs"
FEATURES     = ["RBC","HGB","HCT","MCV","MCH","RDW","WBC","PLT"]
TEST_SIZE    = 0.20
TABNET_ND    = 4               # decision/representation width
EPOCHS       = 25
BATCH_SIZE   = 1024
PATIENCE     = 5
PCA_D4_FAIR  = True
PROP_MARKER  = {"eid":"o","redundancy":"s","skew":"^","missing":"D"}
FILENAME_RE  = re.compile(r"syn_(?P<prop>eid|redundancy|skew|missing)-(?P<level>low|mid|high)_seed(?P<seed>\d+)\.csv")

# shared deterministic logistic label vector
_RNG_LABEL = np.random.default_rng(999)
W_GLOBAL = None

# ---------- official TabNet ----------
from pytorch_tabnet.pretraining import TabNetPretrainer  # official
import torch
from pytorch_tabnet.tab_network import EmbeddingGenerator, TabNetEncoder  # for clear typing (not strictly required)

# ---------- IO ----------
def list_inputs():
    here = Path(__file__).parent
    z = here / DATA_ARCHIVE
    if z.exists():
        with zipfile.ZipFile(z, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv") and n.startswith("syn_")
                     and "/masks/" not in n and "metadata.csv" not in n]
        return ("zip", z, sorted(names))
    d = here / "synthetic_cbc"
    if d.exists() and d.is_dir():
        files = sorted([str(p) for p in d.glob("syn_*.csv")])
        return ("dir", d, files)
    raise FileNotFoundError("Put synthetic_cbc.zip or a 'synthetic_cbc/' folder next to this script.")

def read_csv(kind, container, name):
    if kind == "zip":
        with zipfile.ZipFile(container, "r") as zf:
            with zf.open(name) as f: return pd.read_csv(f)
    else:
        return pd.read_csv(name)

def parse_meta(name):
    base = os.path.basename(name)
    m = FILENAME_RE.match(base)
    if not m: return {"property":"unknown","level":"unknown","seed":0,"stem":base.replace(".csv","")}
    gd = m.groupdict()
    return {"property": gd["prop"], "level": gd["level"], "seed": int(gd["seed"]), "stem": base.replace(".csv","")}

# ---------- labels ----------
def make_rule_label(df_imp):
    hgb = df_imp["HGB"].to_numpy(); pltc = df_imp["PLT"].to_numpy(); wbc = df_imp["WBC"].to_numpy()
    return ((hgb < 10.0) | (pltc < 100.0) | (wbc > 12.0)).astype(np.int32)

def make_logistic_label_global(Xz):
    global W_GLOBAL
    if W_GLOBAL is None:
        W_GLOBAL = _RNG_LABEL.normal(size=Xz.shape[1])
        W_GLOBAL = W_GLOBAL / (np.linalg.norm(W_GLOBAL) + 1e-9)
    s = Xz @ W_GLOBAL
    b = -np.median(s)
    return ((s + b) > 0).astype(np.int32)

# ---------- metrics ----------
def continuity(X, Z, n_neighbors=10):
    n = X.shape[0]; k = min(n_neighbors, n-1)
    nbr_X = NearestNeighbors(n_neighbors=min(k+1, n)).fit(X)
    ind_X = nbr_X.kneighbors(return_distance=False)[:,1:]
    nbr_Z = NearestNeighbors(n_neighbors=n-1).fit(Z)
    ind_Z = nbr_Z.kneighbors(return_distance=False)
    ranks = np.empty((n,n), dtype=np.int32)
    for i in range(n):
        ranks[i, ind_Z[i,:]] = np.arange(1,n); ranks[i,i] = 0
    pen = 0.0
    for i in range(n):
        neighZk = set(ind_Z[i,:k])
        for j in ind_X[i]:
            if j not in neighZk:
                r = ranks[i,j]
                if r>0: pen += (r-k)
    denom = (2.0/(n*k*(2*n-3*k-1))) if (2*n-3*k-1)>0 else 0.0
    return float(1.0 - denom*pen)

def effective_rank(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    S = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    lam = (S**2).astype(np.float64)
    if lam.sum() <= 0: return float("nan")
    p = lam/lam.sum(); H = -(p*np.log(p+1e-12)).sum()
    return float(np.exp(H))

def hubness_skew(Z, k):
    n = Z.shape[0]; k = min(k, n-1)
    inds = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    c = np.bincount(inds.ravel(), minlength=n)
    return 0.0 if c.std()==0 else float(stats.skew(c))

# ---------- extract TabNet encoder representation ----------
@torch.no_grad()
def tabnet_embed_from_pretrainer(pretrainer: TabNetPretrainer, X: np.ndarray) -> np.ndarray:
    """
    Follows TabNet's internal flow: embed -> encoder -> sum(decision steps).
    Matches TabNetNoEmbeddings.forward before the final mapping.
    """
    pretrainer.network.eval()
    device = next(pretrainer.network.parameters()).device
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    embedded = pretrainer.network.embedder(xt)                # (N, post_embed_dim)
    steps_out, _ = pretrainer.network.encoder(embedded)       # list of (N, n_d)
    rep = torch.stack(steps_out, dim=0).sum(dim=0)            # (N, n_d)
    return rep.cpu().numpy()

# ---------- main ----------
def main():
    out = Path(OUTPUT_DIR)
    (out/"plots").mkdir(parents=True, exist_ok=True)
    (out/"tables").mkdir(parents=True, exist_ok=True)

    kind, container, names = list_inputs()
    rows = []

    for name in tqdm(names, desc="TabNet"):
        meta = parse_meta(name); seed = meta["seed"]

        # load & split
        df = read_csv(kind, container, name)
        Xdf = df[FEATURES].copy()
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, shuffle=True)
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=seed, shuffle=True)

        # impute for training + metrics/labels (TabNet cannot ingest NaNs)
        imp = SimpleImputer(strategy="mean")
        Xtr = imp.fit_transform(Xdf.iloc[tr_idx2])
        Xva = imp.transform(Xdf.iloc[va_idx])
        Xte = imp.transform(Xdf.iloc[te_idx])
        # standardize for labels + trust/cont reference space
        scaler = StandardScaler()
        Xtrz = scaler.fit_transform(Xtr); Xtez = scaler.transform(Xte)

        # labels (same as PCA protocol)
        Xallz = scaler.transform(imp.transform(Xdf))
        y_log_all  = make_logistic_label_global(Xallz)
        y_rule_all = make_rule_label(pd.DataFrame(imp.transform(Xdf), columns=FEATURES))
        y_log_tr,  y_log_te  = y_log_all[tr_idx2], y_log_all[te_idx]
        y_rule_tr, y_rule_te = y_rule_all[tr_idx2], y_rule_all[te_idx]

        # ---- Unsupervised pretraining ----
        pre = TabNetPretrainer(
            n_d=TABNET_ND, n_a=TABNET_ND, n_steps=3, gamma=1.3,
            seed=seed, mask_type='sparsemax', device_name='auto'
        )

        # Make sure validation produces at least one batch:
        # use one batch_size that is <= both train and val set sizes and don't drop the last batch
        bs  = min(BATCH_SIZE, Xtr.shape[0], Xva.shape[0])
        vbs = min(128, max(2, bs // 4))

        print(f"[TabNet] Train/Val sizes: {Xtr.shape[0]}/{Xva.shape[0]}  | batch_size={bs}, vbs={vbs}")

        pre.fit(
            X_train=Xtr,
            eval_set=[Xva],              # Pretrainer takes just X for eval_set
            max_epochs=EPOCHS,
            patience=PATIENCE,
            batch_size=bs,               # <= both train and val sizes
            virtual_batch_size=vbs,
            num_workers=0,
            drop_last=False              # <- critical: don't drop the only eval batch
        )


        # ---- Encode (no saving) ----
        Ztr = tabnet_embed_from_pretrainer(pre, Xtr)  # (Ntr, n_d)
        Zte = tabnet_embed_from_pretrainer(pre, Xte)  # (Nte, n_d)

        # variants
        variants = [("TabNet_native", Ztr, Zte)]
        if PCA_D4_FAIR and Ztr.shape[1] != 4:
            p = PCA(n_components=4, random_state=0).fit(Ztr)
            variants.append(("TabNet_d4", p.transform(Ztr), p.transform(Zte)))

        # metrics + utility
        n_test = Xte.shape[0]
        k1 = max(5, int(np.floor(np.sqrt(n_test))))
        k2 = min(n_test-1, max(10, 2*k1))

        for enc_name, Ztr_use, Zte_use in variants:
            trust_k1 = skl_trustworthiness(Xtez, Zte_use, n_neighbors=k1)
            trust_k2 = skl_trustworthiness(Xtez, Zte_use, n_neighbors=k2)
            cont_k1  = continuity(Xtez, Zte_use, n_neighbors=k1)
            cont_k2  = continuity(Xtez, Zte_use, n_neighbors=k2)
            effr     = effective_rank(Zte_use)
            hub_k1   = hubness_skew(Zte_use, k1)
            hub_k2   = hubness_skew(Zte_use, k2)

            def auc_lin(Ztr_, ytr_, Zte_, yte_):
                if len(np.unique(ytr_))<2 or len(np.unique(yte_))<2: return np.nan
                clf = LogisticRegression(max_iter=300, solver="lbfgs")
                clf.fit(Ztr_, ytr_); p = clf.predict_proba(Zte_)[:,1]
                return roc_auc_score(yte_, p)

            def auc_knn(Ztr_, ytr_, Zte_, yte_, k):
                if len(np.unique(ytr_))<2 or len(np.unique(yte_))<2: return np.nan
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(Ztr_, ytr_); p = knn.predict_proba(Zte_)[:,1]
                return roc_auc_score(yte_, p)

            rows.append({
                "file": os.path.basename(name),
                "property": meta["property"], "level": meta["level"], "seed": seed,
                "encoder": enc_name, "n_test": n_test, "k1": k1, "k2": k2,
                "trust_k1": trust_k1, "trust_k2": trust_k2,
                "cont_k1": cont_k1,   "cont_k2": cont_k2,
                "eff_rank": effr, "hubness_k1": hub_k1, "hubness_k2": hub_k2,
                "auc_logistic_lin": auc_lin(Ztr_use, y_log_tr,  Zte_use, y_log_te),
                "auc_rule_lin":     auc_lin(Ztr_use, y_rule_tr, Zte_use, y_rule_te),
                "auc_logistic_knn": auc_knn(Ztr_use, y_log_tr,  Zte_use, y_log_te, k=k1),
                "auc_rule_knn":     auc_knn(Ztr_use, y_rule_tr, Zte_use, y_rule_te, k=k1),
            })

        # cleanup
        del pre, Ztr, Zte, Xtr, Xva, Xte, Xtrz, Xtez
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ----- save results + plots -----
    out = Path(OUTPUT_DIR); out.mkdir(exist_ok=True, parents=True)
    dfres = pd.DataFrame(rows)
    dfres.to_csv(out/"results_tabnet8F4D.csv", index=False)

    plots = out/"plots"; plots.mkdir(parents=True, exist_ok=True)
    for enc in sorted(dfres["encoder"].unique()):
        sub = dfres[dfres["encoder"]==enc]
        plt.figure(figsize=(9,5.5))
        for prop, m in PROP_MARKER.items():
            s = sub[sub["property"]==prop]
            if s.empty: continue
            plt.scatter(s["eff_rank"], s["auc_logistic_knn"], marker=m, s=36, alpha=0.9, label=prop)
        plt.xlabel("Effective Rank"); plt.ylabel("AUC (k-NN, logistic)")
        plt.title(f"Intrinsic → Utility — {enc}")
        plt.legend(title="Property", ncols=2); plt.tight_layout()
        plt.savefig(plots/f"fm_metric_to_auc_{enc}.png", dpi=150); plt.close()

    tables = out/"tables"; tables.mkdir(parents=True, exist_ok=True)
    dfres.loc[:, ["file","property","level","seed","encoder","eff_rank","auc_logistic_knn"]].to_csv(
        tables/"tabnet_figureB_points.csv", index=False
    )
    print(f"[OK] Wrote {out/'results_tabnet.csv'} with {len(dfres)} rows.")

if __name__ == "__main__":
    main()
