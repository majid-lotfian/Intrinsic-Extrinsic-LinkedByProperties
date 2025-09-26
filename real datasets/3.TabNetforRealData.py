#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TabNet on real datasets (embeddings at 4D, 16D, 32D)

- Data: CBC.csv, Covid19.csv, Iraq.xlsx, Liverpool.csv (placed next to this script)
- IO: CSV/XLSX reader, header cleaning, sex-> {0,1}, keep age, drop id from features, drop *date* cols, numeric-only
- Split: 80/20 (seed=42); 10% of train -> validation (early stopping)
- TabNet: Unsupervised pretraining (TabNetPretrainer), then build TabNetClassifier from the unsupervised checkpoint
          Embedding = n_d representation (we use n_steps=1 so embedding dim == n_d)
- Metrics: trustworthiness, continuity, effective rank, hubness (on test)
- Utility: AUC (Logistic + kNN@k1) on embeddings for logistic label (always) and rule label (if synonyms exist)
- Property stats: ER_norm/avg|corr|/mean|skew|/missing_rate + low/mid/high via thresholds from 24F synthetic metadata.csv

Outputs under TabNetReal/:
  results_tabnet.csv                         # per dataset × hidden_dim
  embeddings/<dataset>_<dim>d_train.npz
  embeddings/<dataset>_<dim>d_test.npz
  models/<dataset>_<dim>d/                   # pretrainer + classifier states
  tables/real_property_stats.csv
  tables/real_property_levels.csv
  tables/thresholds_used.csv
  plots/er_to_auc_4d.png
  plots/er_to_auc_16d.png
  plots/er_to_auc_32d.png
"""

import warnings, gc, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import trustworthiness as skl_trustworthiness
import matplotlib.pyplot as plt
import torch

# pytorch-tabnet
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
OUTPUT_DIR = "TabNetReal"
DATASETS = ["CBC.csv", "Covid19.csv", "Iraq.xlsx", "Liverpool.csv"]
HIDDEN_DIMS = [4, 16, 32]  # embeddings to produce (n_d)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- IO & header cleaning ----------

def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Clean column names: strip quotes/space, collapse whitespace -> _
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.strip('"').str.strip("'")
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def encode_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    m = s.isin(["m","male","1","true","t"])
    f = s.isin(["f","female","0","false","f"])
    return pd.Series(np.where(m, 1, np.where(f, 0, np.nan)), index=series.index, dtype="float64")

def load_dataset(path: Path):
    df = _read_table(path)
    # keep id for reference
    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    ids = df[id_col].astype("Int64") if id_col else pd.Series(np.arange(1, len(df)+1, dtype=np.int64), name="id")
    # encode sex/gender
    for c in df.columns:
        if c.lower() in ["sex", "gender"]:
            df[c] = encode_sex(df[c])
    # drop date-like columns
    drop_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")
    # numeric-only; drop id from features
    num_df = df.select_dtypes(include=[np.number]).copy()
    if id_col and id_col in num_df.columns:
        num_df = num_df.drop(columns=[id_col])
    return df, num_df, ids

# ---------- Rule label synonyms ----------

WBC_ALIASES = [
    "white_blood_cells", "leukocyte_count",
    "white_blood_cell_count", "white_blood_cell", "White_Blood_Cell"
]
PLT_ALIASES = ["platelets", "platelet_count", "Platelet"]
HGB_ALIASES = ["hemoglobin", "Hemoglobin"]

def normalize_rule_columns(df: pd.DataFrame):
    cols = {"white_blood_cells": None, "platelets": None, "hemoglobin": None}
    lower = {c.lower(): c for c in df.columns}
    def find_alias(aliases):
        for a in aliases:
            a_low = a.lower()
            if a_low in lower:
                return lower[a_low]
        return None
    cols["white_blood_cells"] = find_alias(WBC_ALIASES)
    cols["platelets"] = find_alias(PLT_ALIASES)
    cols["hemoglobin"] = find_alias(HGB_ALIASES)
    return cols

def make_rule_label(df: pd.DataFrame, name_map) -> np.ndarray | None:
    w = name_map["white_blood_cells"]; p = name_map["platelets"]; h = name_map["hemoglobin"]
    if any(v is None for v in [w, p, h]): return None
    wbc = df[w].to_numpy()
    pltc = df[p].to_numpy()
    hgb = df[h].to_numpy()
    return ((hgb < 10.0) | (pltc < 100.0) | (wbc > 12.0)).astype(np.int32)

# ---------- Intrinsic metrics ----------

def continuity(X, Z, n_neighbors=10):
    n = X.shape[0]; k = min(n_neighbors, n-1)
    nbr_X = NearestNeighbors(n_neighbors=min(k+1, n)).fit(X)
    ind_X = nbr_X.kneighbors(return_distance=False)[:,1:]
    nbr_Z = NearestNeighbors(n_neighbors=n-1).fit(Z)
    ind_Z = nbr_Z.kneighbors(return_distance=False)
    ranks = np.empty((n,n), dtype=np.int32)
    for i in range(n):
        ranks[i, ind_Z[i,:]] = np.arange(1,n)
        ranks[i,i] = 0
    pen = 0.0
    for i in range(n):
        neighZk = set(ind_Z[i,:k])
        for j in ind_X[i]:
            if j not in neighZk:
                r = ranks[i,j]
                if r>0: pen += (r-k)
    denom = (2.0/(n*k*(2*n-3*k-1))) if (2*n-3*k-1) > 0 else 0.0
    return float(1.0 - denom*pen)

def effective_rank(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    S = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    lam = (S**2).astype(np.float64)
    if lam.sum() <= 0: return float("nan")
    p = lam/lam.sum()
    H = -(p*np.log(p+1e-12)).sum()
    return float(np.exp(H))

def hubness_skew(Z, k):
    n = Z.shape[0]; k = min(k, n-1)
    inds = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    c = np.bincount(inds.ravel(), minlength=n)
    return 0.0 if c.std()==0 else float(stats.skew(c))

# ---------- Logistic label (dimension-safe) ----------

_W_CACHE: dict[int, np.ndarray] = {}

def make_logistic_label_global(Xz: np.ndarray) -> np.ndarray:
    """
    Deterministic synthetic label per input feature dimension (d = Xz.shape[1]),
    balanced via median threshold.
    """
    d = Xz.shape[1]
    w = _W_CACHE.get(d)
    if w is None:
        rng = np.random.default_rng(999 + d)
        w = rng.normal(size=d)
        w = w / (np.linalg.norm(w) + 1e-9)
        _W_CACHE[d] = w
    s = Xz @ w
    b = -np.median(s)
    return (s + b > 0).astype(np.int32)

# ---------- Property stats & thresholds (from synthetic metadata) ----------

def effective_rank_from_corr(R: np.ndarray) -> float:
    vals = np.linalg.eigvalsh((R + R.T) / 2.0)
    vals = np.clip(vals, 1e-12, None)
    p = vals / vals.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))

def avg_abs_corr(R: np.ndarray) -> float:
    d = R.shape[0]
    mask = ~np.eye(d, dtype=bool)
    return float(np.mean(np.abs(R[mask])))

def mean_abs_skew_matrix(X: np.ndarray) -> float:
    vals = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 3:
            vals.append(stats.skew(col))
    return float(np.mean(np.abs(vals))) if vals else float("nan")

def dataset_property_stats(df_num: pd.DataFrame) -> dict:
    X = df_num.to_numpy(dtype=float)
    miss_rate = float(np.mean(~np.isfinite(X)))
    X_complete = X[np.all(np.isfinite(X), axis=1)]
    d = X.shape[1]
    if X_complete.shape[0] >= max(30, d+5):
        Xz = (X_complete - np.nanmean(X_complete, axis=0)) / (np.nanstd(X_complete, axis=0) + 1e-9)
        R = np.corrcoef(Xz, rowvar=False)
        er = effective_rank_from_corr(R)
        er_norm = er / d
        rbar = avg_abs_corr(R)
    else:
        er = er_norm = rbar = float("nan")
    skew_abs = mean_abs_skew_matrix(X)
    return dict(er=er, er_norm=er_norm, redundancy=rbar, skew_abs=skew_abs, missing_rate=miss_rate, d=d)

def load_thresholds_from_metadata(meta_path: Path, d_synth: int = 24) -> dict:
    meta = pd.read_csv(meta_path)
    def midpoints(series_low, series_mid, series_high):
        m_low = np.nanmedian(series_low); m_mid = np.nanmedian(series_mid); m_high = np.nanmedian(series_high)
        t1 = 0.5*(m_low + m_mid); t2 = 0.5*(m_mid + m_high)
        return float(t1), float(t2)
    cuts = {}
    sdf = meta[meta["property"]=="eid"].copy()
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_eff_rank"]/d_synth,
                       sdf[sdf["level"]=="mid"]["realized_eff_rank"]/d_synth,
                       sdf[sdf["level"]=="high"]["realized_eff_rank"]/d_synth)
    cuts["eid"] = (t1, t2)
    sdf = meta[meta["property"]=="redundancy"]
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_avg_abs_corr"],
                       sdf[sdf["level"]=="mid"]["realized_avg_abs_corr"],
                       sdf[sdf["level"]=="high"]["realized_avg_abs_corr"])
    cuts["redundancy"] = (t1, t2)
    sdf = meta[meta["property"]=="skew"]
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_mean_abs_skew"],
                       sdf[sdf["level"]=="mid"]["realized_mean_abs_skew"],
                       sdf[sdf["level"]=="high"]["realized_mean_abs_skew"])
    cuts["skew"] = (t1, t2)
    sdf = meta[meta["property"]=="missing"]
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_missing_rate"],
                       sdf[sdf["level"]=="mid"]["realized_missing_rate"],
                       sdf[sdf["level"]=="high"]["realized_missing_rate"])
    cuts["missing"] = (t1, t2)
    return cuts

def classify_levels(stats: dict, cuts: dict) -> dict:
    def bin3(x, t12):
        t1, t2 = t12
        if not np.isfinite(x): return "unknown"
        return "low" if x < t1 else ("mid" if x < t2 else "high")
    return {
        "eid_level": bin3(stats["er_norm"], cuts["eid"]),
        "redundancy_level": bin3(stats["redundancy"], cuts["redundancy"]),
        "skew_level": bin3(stats["skew_abs"], cuts["skew"]),
        "missing_level": bin3(stats["missing_rate"], cuts["missing"]),
    }

# ---------- TabNet embedding extraction (n_d features) ----------

@torch.no_grad()
def tabnet_encode_classifier(clf: TabNetClassifier, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Extract n_d-dimensional representation from TabNetClassifier.network.
    We set n_steps=1 so the network's output before final mapping is (B, n_d).
    """
    clf.network.eval()
    reps = []
    # move network to device once
    device = next(clf.network.parameters()).device
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
        out = clf.network(xb)
        # Known library behaviors:
        # - Some versions return the representation tensor directly.
        # - Others return a tuple where the first element is the representation.
        if isinstance(out, (list, tuple)):
            z = out[0]
        else:
            z = out
        reps.append(z.detach().cpu().numpy())
    return np.concatenate(reps, axis=0)

# ---------- Main ----------

def main():
    here = Path(__file__).parent
    out = here / OUTPUT_DIR
    (out/"embeddings").mkdir(parents=True, exist_ok=True)
    (out/"models").mkdir(parents=True, exist_ok=True)
    (out/"tables").mkdir(parents=True, exist_ok=True)
    (out/"plots").mkdir(parents=True, exist_ok=True)

    # thresholds from synthetic (24F)
    meta_path = here / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.csv (24-feature synthetic metadata) not found next to this script.")
    cuts = load_thresholds_from_metadata(meta_path, d_synth=24)
    pd.DataFrame([
        {"property":"eid", "cut_low_mid":cuts["eid"][0], "cut_mid_high":cuts["eid"][1], "normalized":"ER_norm"},
        {"property":"redundancy", "cut_low_mid":cuts["redundancy"][0], "cut_mid_high":cuts["redundancy"][1]},
        {"property":"skew", "cut_low_mid":cuts["skew"][0], "cut_mid_high":cuts["skew"][1]},
        {"property":"missing", "cut_low_mid":cuts["missing"][0], "cut_mid_high":cuts["missing"][1]},
    ]).to_csv(out/"tables/thresholds_used.csv", index=False)

    all_rows = []
    prop_rows = []

    for fname in DATASETS:
        path = here / fname
        if not path.exists():
            print(f"[WARN] Missing dataset: {fname}; skipping.")
            continue

        print(f"[INFO] Loading {fname}")
        df_raw, df_num, ids_all = load_dataset(path)
        stem = path.stem

        # property stats/levels ON RAW NUMERIC FEATURES
        stats_dict = dataset_property_stats(df_num)
        levels = classify_levels(stats_dict, cuts)

        # rule columns map
        rule_map = normalize_rule_columns(df_raw)

        # split
        idx = np.arange(len(df_num))
        tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)

        # one split to get train2/val; ensure at least 1 val example
        n_tr  = len(tr_idx)
        n_val = max(1, int(np.ceil(0.10 * n_tr)))
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=n_val, random_state=SEED, shuffle=True)

        # Impute/scale for labels & trust/cont metrics
        imp = SimpleImputer(strategy="mean")
        Xtr  = imp.fit_transform(df_num.iloc[tr_idx])     # full train (used to fit scaler)
        Xtr2 = imp.transform(df_num.iloc[tr_idx2])        # <-- train2 subset (used everywhere else)
        Xva  = imp.transform(df_num.iloc[va_idx])
        Xte  = imp.transform(df_num.iloc[te_idx])
        Xall = imp.transform(df_num)

        scaler = StandardScaler()
        Xtrz  = scaler.fit_transform(Xtr)                 # fit scaler on full train (labels metrics)
        Xtez  = scaler.transform(Xte)
        Xallz = scaler.transform(Xall)

        # labels
        y_log_all = make_logistic_label_global(Xallz)
        y_log_tr2, y_log_te = y_log_all[tr_idx2], y_log_all[te_idx]

        y_rule_all = make_rule_label(df_raw, rule_map)
        if y_rule_all is not None:
            y_rule_tr2, y_rule_te = y_rule_all[tr_idx2], y_rule_all[te_idx]
        else:
            y_rule_tr2 = y_rule_te = None

        # For TabNet training/encoding: use mean-imputed (not z-scored) arrays
        Xtr2_tab = Xtr2.astype(np.float32)                # <-- use train2
        Xva_tab  = Xva.astype(np.float32)
        Xte_tab  = Xte.astype(np.float32)

        # --- NEW: guard for small datasets ---
        has_val = Xva_tab.shape[0] > 0
        if has_val:
            bs  = max(8, min(256, Xtr2_tab.shape[0], Xva_tab.shape[0]))  # <= both train2 & val
        else:
            bs  = max(8, min(256, Xtr2_tab.shape[0]))
        vbs = max(8, min(64, bs // 2))

        n_test = Xtez.shape[0]
        k1 = max(5, int(math.floor(math.sqrt(n_test))))
        k2 = min(n_test-1, max(10, 2*k1))

        # Train TabNet per hidden dim (embedding dim = n_d)
        for dim in HIDDEN_DIMS:
            print(f"[INFO] {stem}: TabNet pretrain (n_d={dim}, n_steps=1) on {DEVICE}")

            # Unsupervised pretraining
            pre = TabNetPretrainer(
                n_d=dim, n_a=dim,
                n_steps=1,               # keep 1 step so representation is exactly n_d
                gamma=1.3,
                n_independent=2, n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-3),
                mask_type='entmax',
                seed=SEED,
                verbose=0,
                device_name=DEVICE
            )
            print(f"[DEBUG] train2={Xtr2_tab.shape[0]}, val={Xva_tab.shape[0]}, bs={bs}, vbs={vbs}, has_val={has_val}")


            pre.fit(
                X_train=Xtr2_tab,
                eval_set=[Xva_tab] if has_val else None,   # <-- only if we have val,
                max_epochs=200,
                patience=30 if has_val else 0,             # <-- disable early stop if no val,
                batch_size=bs,
                virtual_batch_size=vbs
            )

            # Build classifier (no supervised fine-tune) just to reuse the encoder
            clf = TabNetClassifier(
                n_d=dim, n_a=dim,
                n_steps=1,
                gamma=1.3,
                n_independent=2, n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-3),
                mask_type='entmax',
                seed=SEED,
                verbose=0,
                device_name=DEVICE
            )

            # Initialize the classifier’s network AND load unsupervised weights
            # without doing any meaningful supervised training.

            y_dummy_tr2 = np.zeros(Xtr2_tab.shape[0], dtype=int)

            if has_val:
                y_dummy_va = np.zeros(Xva_tab.shape[0], dtype=int)
                clf.fit(
                    X_train=Xtr2_tab, y_train=y_dummy_tr2,
                    eval_set=[(Xva_tab, y_dummy_va)],
                    max_epochs=1, patience=1,
                    batch_size=bs, virtual_batch_size=vbs,
                    from_unsupervised=pre
                )
            else:
                clf.fit(
                    X_train=Xtr2_tab, y_train=y_dummy_tr2,
                    max_epochs=1, patience=1,
                    batch_size=bs, virtual_batch_size=vbs,
                    from_unsupervised=pre
                )


            #y_dummy_va = np.zeros(Xva_tab.shape[0], dtype=int)



            #clf.load_weights_from_unsupervised(pre)

            # Encode (representation z in R^{n_d})
            Ztr = tabnet_encode_classifier(clf, Xtr2_tab)
            Zte = tabnet_encode_classifier(clf, Xte_tab)

            # save embeddings & models
            ckpt_dir = out/"models"/f"{stem}_{dim}d"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            pre.save_model(str(ckpt_dir/ "pretrainer"))
            clf.save_model(str(ckpt_dir/ "classifier"))

            emb_tr = out/"embeddings"/f"{stem}_{dim}d_train.npz"
            emb_te = out/"embeddings"/f"{stem}_{dim}d_test.npz"
            np.savez_compressed(emb_tr, id=ids_all.iloc[tr_idx2].to_numpy(dtype=np.int64), Z=Ztr.astype(np.float32))
            np.savez_compressed(emb_te, id=ids_all.iloc[te_idx].to_numpy(dtype=np.int64), Z=Zte.astype(np.float32))

            # intrinsic metrics on test
            trust_k1 = skl_trustworthiness(Xtez, Zte, n_neighbors=k1)
            trust_k2 = skl_trustworthiness(Xtez, Zte, n_neighbors=k2)
            cont_k1  = continuity(Xtez, Zte, n_neighbors=k1)
            cont_k2  = continuity(Xtez, Zte, n_neighbors=k2)
            effr     = effective_rank(Zte)
            hub_k1   = hubness_skew(Zte, k1)
            hub_k2   = hubness_skew(Zte, k2)

            # utility (on embeddings)
            def auc_lin(Ztr_, ytr_, Zte_, yte_):
                if len(np.unique(ytr_)) < 2 or len(np.unique(yte_)) < 2: return np.nan
                clf_ = LogisticRegression(max_iter=300, solver="lbfgs")
                clf_.fit(Ztr_, ytr_); p = clf_.predict_proba(Zte_)[:,1]
                return roc_auc_score(yte_, p)

            def auc_knn(Ztr_, ytr_, Zte_, yte_, k):
                if len(np.unique(ytr_)) < 2 or len(np.unique(yte_)) < 2: return np.nan
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(Ztr_, ytr_); p = knn.predict_proba(Zte_)[:,1]
                return roc_auc_score(yte_, p)

            auc_log_lin  = auc_lin(Ztr, y_log_tr2,  Zte, y_log_te)
            auc_log_knn  = auc_knn(Ztr, y_log_tr2, Zte, y_log_te, k=k1)

            if y_rule_tr2 is not None:
                auc_rule_lin = auc_lin(Ztr, y_rule_tr2, Zte, y_rule_te)
                auc_rule_knn = auc_knn(Ztr, y_rule_tr2, Zte, y_rule_te, k=k1)
            else:
                auc_rule_lin = np.nan
                auc_rule_knn = np.nan

            all_rows.append({
                "dataset": stem,
                "encoder": "TabNet",
                "hidden_dim": dim,
                "d_features": df_num.shape[1],
                "n_total": df_num.shape[0],
                "n_test": n_test, "k1": k1, "k2": k2,
                "trust_k1": trust_k1, "trust_k2": trust_k2,
                "cont_k1": cont_k1,   "cont_k2": cont_k2,
                "eff_rank": effr, "hubness_k1": hub_k1, "hubness_k2": hub_k2,
                "auc_logistic_lin": auc_log_lin, "auc_logistic_knn": auc_log_knn,
                "auc_rule_lin": auc_rule_lin, "auc_rule_knn": auc_rule_knn,
                "emb_train_path": str(emb_tr), "emb_test_path": str(emb_te),
                "ckpt_dir": str(ckpt_dir),
            })

            # cleanup GPU memory between dims
            del pre, clf, Ztr, Zte
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            gc.collect()

        # record property stats once per dataset
        prop_rows.append({
            "dataset": stem,
            "d_features": stats_dict["d"],
            "er": stats_dict["er"],
            "er_norm": stats_dict["er_norm"],
            "avg_abs_corr": stats_dict["redundancy"],
            "mean_abs_skew": stats_dict["skew_abs"],
            "missing_rate": stats_dict["missing_rate"],
            **levels
        })

    # save results & tables
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out/"results_tabnet.csv", index=False)

    prop_df = pd.DataFrame(prop_rows)
    prop_df.to_csv(out/"tables"/"real_property_stats.csv", index=False)
    prop_df.to_csv(out/"tables"/"real_property_levels.csv", index=False)

    # plots: ER → AUC (logistic kNN) per dim
    if not out_df.empty:
        for dim in sorted(out_df["hidden_dim"].unique()):
            sub = out_df[out_df["hidden_dim"] == dim]
            plt.figure(figsize=(8,5))
            for ds, grp in sub.groupby("dataset"):
                plt.scatter(grp["eff_rank"], grp["auc_logistic_knn"], label=ds, s=48, alpha=0.9)
            plt.xlabel("Effective Rank (embedding)")
            plt.ylabel("AUC (k-NN, logistic label)")
            plt.title(f"Intrinsic (Effective Rank) → Utility (k-NN AUC) — TabNet {dim}D")
            plt.legend(title="Dataset")
            plt.tight_layout()
            plt.savefig(out/"plots"/f"er_to_auc_{dim}d.png", dpi=150)
            plt.close()

    print(f"[OK] Wrote {out/'results_tabnet.csv'} with {len(out_df)} rows.")

if __name__ == "__main__":
    main()
