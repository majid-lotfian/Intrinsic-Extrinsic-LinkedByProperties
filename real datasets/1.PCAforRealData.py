#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-data PCA pipeline (4 datasets):
- Reads 4 CSVs in the same folder as this script: CBC.csv, Covid19.csv, Iraq.csv, Liverpool.csv
- 80/20 split (seed=42), mean-impute (train), z-score (train), PCA(d=4), save embeddings (.npz)
- Metrics (test): Trustworthiness, Continuity, Effective Rank, Hubness
- Utility (test): Linear AUC + k-NN AUC for (a) logistic label, (b) rule label if available
- Property stats per dataset (ER_norm, avg|corr|, mean|skew|, missing rate)
- Classifies each dataset's property level (low/mid/high) using thresholds derived from synthetic metadata (24F)

Outputs:
    outputsReal/results_pca.csv
    outputsReal/embeddings/<dataset>_train.npz, <dataset>_test.npz
    outputsReal/models/<dataset>_pca_model.npz
    outputsReal/tables/real_property_levels.csv
    outputsReal/tables/real_property_stats.csv
    outputsReal/tables/thresholds_used.csv            # copy of loaded thresholds
    outputsReal/plots/fig_metric_to_auc_pca.png       # ER → AUC scatter (dataset markers)

Usage:
    Place CBC.csv, Covid19.csv, Iraq.csv, Liverpool.csv next to this script.
    Also place the synthetic metadata file: metadata.csv (24-feature synthetic runs).
    Run: python real_pca_pipeline.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness as skl_trustworthiness
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
DATASETS = ["CBC.csv", "Covid19.csv", "Iraq.xlsx", "Liverpool.csv"]
OUTPUT_DIR = "outputsReal"
D_EMB = 4


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        # needs: pip install openpyxl
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Clean weird headers (quotes, tabs/spaces -> underscores)
    df.columns = (
        df.columns.astype(str)
        .str.strip()              # trim whitespace
        .str.strip('"').str.strip("'")  # drop stray quotes
        .str.replace(r"\s+", "_", regex=True)  # spaces/tabs -> _
    )
    return df


# Column normalization for rule label
WBC_ALIASES = [
    "white_blood_cells", "leukocyte_count",
    "white_blood_cell_count", "white_blood_cell", "White_Blood_Cell"
]
PLT_ALIASES = ["platelets", "platelet_count", "Platelet"]
HGB_ALIASES = ["hemoglobin", "Hemoglobin"]


def normalize_rule_columns(df):
    cols = { "white_blood_cells": None, "platelets": None, "hemoglobin": None }
    lower = {c.lower(): c for c in df.columns}
    def find_alias(aliases):
        for a in aliases:
            a_low = a.lower()
            if a_low in lower: return lower[a_low]
        return None
    cols["white_blood_cells"] = find_alias(WBC_ALIASES)
    cols["platelets"] = find_alias(PLT_ALIASES)
    cols["hemoglobin"] = find_alias(HGB_ALIASES)
    return cols

# Intrinsic metrics
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

# Labels
# Dimension-aware, deterministic logistic label
_W_CACHE = {}  # maps d -> weight vector

def make_logistic_label_global(Xz):
    """
    Creates a deterministic synthetic logistic label via a fixed random
    projection per feature dimension (d = Xz.shape[1]).
    The bias centers the scores so labels are ~balanced per dataset.
    """
    d = Xz.shape[1]
    w = _W_CACHE.get(d)
    if w is None:
        rng = np.random.default_rng(999 + d)  # deterministic per dimension
        w = rng.normal(size=d)
        w = w / (np.linalg.norm(w) + 1e-9)
        _W_CACHE[d] = w
    s = Xz @ w
    b = -np.median(s)
    return (s + b > 0).astype(np.int32)

def make_rule_label(df, name_map):
    w = name_map["white_blood_cells"]; p = name_map["platelets"]; h = name_map["hemoglobin"]
    if any(v is None for v in [w,p,h]): return None  # cannot compute
    wbc = df[w].to_numpy()
    pltc = df[p].to_numpy()
    hgb = df[h].to_numpy()
    return ((hgb < 10.0) | (pltc < 100.0) | (wbc > 12.0)).astype(np.int32)

# Property stats for low/mid/high classification
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
    # EID (normalize by d_synth)
    sdf = meta[meta["property"]=="eid"].copy()
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_eff_rank"]/d_synth,
                       sdf[sdf["level"]=="mid"]["realized_eff_rank"]/d_synth,
                       sdf[sdf["level"]=="high"]["realized_eff_rank"]/d_synth)
    cuts["eid"] = (t1, t2)
    # Redundancy
    sdf = meta[meta["property"]=="redundancy"]
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_avg_abs_corr"],
                       sdf[sdf["level"]=="mid"]["realized_avg_abs_corr"],
                       sdf[sdf["level"]=="high"]["realized_avg_abs_corr"])
    cuts["redundancy"] = (t1, t2)
    # Skew
    sdf = meta[meta["property"]=="skew"]
    t1, t2 = midpoints(sdf[sdf["level"]=="low"]["realized_mean_abs_skew"],
                       sdf[sdf["level"]=="mid"]["realized_mean_abs_skew"],
                       sdf[sdf["level"]=="high"]["realized_mean_abs_skew"])
    cuts["skew"] = (t1, t2)
    # Missing
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

def encode_sex(series):
    # Map common encodings to {0,1}; return numeric with NaN for unknowns
    s = series.astype(str).str.strip().str.lower()
    m = s.isin(["m","male","1","true","t"])
    f = s.isin(["f","female","0","false","f"])
    out = pd.Series(np.where(m, 1, np.where(f, 0, np.nan)), index=series.index, dtype="float64")
    return out

def load_dataset(path: Path):
    df = _read_table(path)

    # Keep a copy of ids if present
    id_col = None
    for c in df.columns:
        if c.lower() == "id":
            id_col = c
            break
    ids = (
        df[id_col].astype("Int64")
        if id_col else
        pd.Series(np.arange(1, len(df)+1, dtype=np.int64), name="id")
    )

    # Encode sex/gender if present
    for c in df.columns:
        if c.lower() in ["sex","gender"]:
            df[c] = encode_sex(df[c])

    # Drop date-like columns
    drop_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Keep numeric features only; drop id from features
    num_df = df.select_dtypes(include=[np.number]).copy()
    if id_col and id_col in num_df.columns:
        num_df = num_df.drop(columns=[id_col])

    return df, num_df, ids


def main():
    here = Path(__file__).parent
    out = here / OUTPUT_DIR
    (out/"embeddings").mkdir(parents=True, exist_ok=True)
    (out/"models").mkdir(parents=True, exist_ok=True)
    (out/"tables").mkdir(parents=True, exist_ok=True)
    (out/"plots").mkdir(parents=True, exist_ok=True)

    # Load thresholds from synthetic metadata (24F)
    meta_path = here / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("Expected metadata.csv (24-feature synthetic metadata) next to the script.")
    cuts = load_thresholds_from_metadata(meta_path, d_synth=24)
    # Save thresholds used
    pd.DataFrame([
        {"property":"eid", "cut_low_mid":cuts["eid"][0], "cut_mid_high":cuts["eid"][1], "normalized":"ER_norm"},
        {"property":"redundancy", "cut_low_mid":cuts["redundancy"][0], "cut_mid_high":cuts["redundancy"][1]},
        {"property":"skew", "cut_low_mid":cuts["skew"][0], "cut_mid_high":cuts["skew"][1]},
        {"property":"missing", "cut_low_mid":cuts["missing"][0], "cut_mid_high":cuts["missing"][1]},
    ]).to_csv(out/"tables/thresholds_used.csv", index=False)

    rows = []
    prop_stats_rows = []
    for fname in DATASETS:
        path = here / fname
        if not path.exists():
            print(f"[WARN] Missing dataset: {fname}; skipping.")
            continue

        print(f"[INFO] Loading {fname}")
        df_raw, df_num, ids_all = load_dataset(path)
        # Rule columns mapping
        rule_map = normalize_rule_columns(df_raw)

        # Split
        idx = np.arange(len(df_num))
        tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)

        # Impute & scale
        imp = SimpleImputer(strategy="mean")
        Xtr  = imp.fit_transform(df_num.iloc[tr_idx])
        Xte  = imp.transform(df_num.iloc[te_idx])
        Xall = imp.transform(df_num)

        scaler = StandardScaler()
        Xtrz = scaler.fit_transform(Xtr); Xtez = scaler.transform(Xte); Xallz = scaler.transform(Xall)

        #global W_GLOBAL
        #W_GLOBAL = None

        # Labels
        y_log_all = make_logistic_label_global(Xallz)
        y_log_tr, y_log_te = y_log_all[tr_idx], y_log_all[te_idx]

        y_rule_all = make_rule_label(df_raw, rule_map)
        if y_rule_all is not None:
            y_rule_tr, y_rule_te = y_rule_all[tr_idx], y_rule_all[te_idx]
        else:
            y_rule_tr = y_rule_te = None

        # PCA(d=4)
        pca = PCA(n_components=D_EMB, random_state=SEED)
        Ztr = pca.fit_transform(Xtrz); Zte = pca.transform(Xtez)

        # Save embeddings + model params
        stem = Path(fname).stem
        emb_tr = out/"embeddings"/f"{stem}_train.npz"
        emb_te = out/"embeddings"/f"{stem}_test.npz"
        np.savez_compressed(emb_tr, id=ids_all.iloc[tr_idx].to_numpy(dtype=np.int64), Z=Ztr.astype(np.float32))
        np.savez_compressed(emb_te, id=ids_all.iloc[te_idx].to_numpy(dtype=np.int64), Z=Zte.astype(np.float32))
        np.savez_compressed(out/"models"/f"{stem}_pca_model.npz",
                            feature_names=df_num.columns.to_numpy(object),
                            imputer_statistics_=imp.statistics_,
                            scaler_mean_=scaler.mean_, scaler_scale_=scaler.scale_,
                            pca_components_=pca.components_, pca_mean_=pca.mean_,
                            pca_explained_variance_ratio_=pca.explained_variance_ratio_,
                            pca_singular_values_=pca.singular_values_, d_emb=np.array([D_EMB]),
                            seed=np.array([SEED]))

        # Metrics on test
        n_test = Xtez.shape[0]
        k1 = max(5, int(np.floor(np.sqrt(n_test))))
        k2 = min(n_test-1, max(10, 2*k1))
        trust_k1 = skl_trustworthiness(Xtez, Zte, n_neighbors=k1)
        trust_k2 = skl_trustworthiness(Xtez, Zte, n_neighbors=k2)
        cont_k1  = continuity(Xtez, Zte, n_neighbors=k1)
        cont_k2  = continuity(Xtez, Zte, n_neighbors=k2)
        effr     = effective_rank(Zte)
        hub_k1   = hubness_skew(Zte, k1)
        hub_k2   = hubness_skew(Zte, k2)

        # Utility
        def auc_lin(Ztr, ytr, Zte, yte):
            if len(np.unique(ytr))<2 or len(np.unique(yte))<2: return np.nan
            clf = LogisticRegression(max_iter=300, solver="lbfgs")
            clf.fit(Ztr, ytr); p = clf.predict_proba(Zte)[:,1]
            return roc_auc_score(yte, p)
        def auc_knn(Ztr, ytr, Zte, yte, k):
            if len(np.unique(ytr))<2 or len(np.unique(yte))<2: return np.nan
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(Ztr, ytr); p = knn.predict_proba(Zte)[:,1]
            return roc_auc_score(yte, p)

        auc_log_lin  = auc_lin(Ztr, y_log_tr,  Zte, y_log_te)
        auc_log_knn  = auc_knn(Ztr, y_log_tr,  Zte, y_log_te, k=k1)

        if y_rule_tr is not None:
            auc_rule_lin = auc_lin(Ztr, y_rule_tr, Zte, y_rule_te)
            auc_rule_knn = auc_knn(Ztr, y_rule_tr, Zte, y_rule_te, k=k1)
        else:
            auc_rule_lin = np.nan
            auc_rule_knn = np.nan

        # Property stats & levels
        stats_dict = dataset_property_stats(df_num)
        levels = classify_levels(stats_dict, cuts)
        prop_stats_rows.append({
            "dataset": Path(fname).stem,
            "d_features": stats_dict["d"],
            "er": stats_dict["er"],
            "er_norm": stats_dict["er_norm"],
            "avg_abs_corr": stats_dict["redundancy"],
            "mean_abs_skew": stats_dict["skew_abs"],
            "missing_rate": stats_dict["missing_rate"],
            **levels
        })

        rows.append({
            "dataset": Path(fname).stem,
            "encoder": "PCA",
            "d_features": df_num.shape[1],
            "n_total": df_num.shape[0],
            "n_test": n_test, "k1": k1, "k2": k2,
            "trust_k1": trust_k1, "trust_k2": trust_k2,
            "cont_k1": cont_k1,   "cont_k2": cont_k2,
            "eff_rank": effr, "hubness_k1": hub_k1, "hubness_k2": hub_k2,
            "auc_logistic_lin": auc_log_lin, "auc_logistic_knn": auc_log_knn,
            "auc_rule_lin": auc_rule_lin, "auc_rule_knn": auc_rule_knn,
            "emb_train_path": str(emb_tr), "emb_test_path": str(emb_te)
        })

    # Save results
    res = pd.DataFrame(rows)
    res_path = out/"results_pca.csv"
    res.to_csv(res_path, index=False)
    print(f"[OK] Wrote {res_path} with {len(res)} rows.")

    prop_stats_df = pd.DataFrame(prop_stats_rows)
    prop_stats_df.to_csv(out/"tables/real_property_stats.csv", index=False)
    prop_stats_df.to_csv(out/"tables/real_property_levels.csv", index=False)  # includes levels

    # Plot: ER → AUC (logistic, k-NN), marker by dataset
    if not res.empty:
        plt.figure(figsize=(8,5))
        for ds, sub in res.groupby("dataset"):
            plt.scatter(sub["eff_rank"], sub["auc_logistic_knn"], label=ds, s=48, alpha=0.9)
        plt.xlabel("Effective Rank (embedding)")
        plt.ylabel("AUC (k-NN, logistic label)")
        plt.title("Intrinsic (Effective Rank) → Utility (k-NN AUC)")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig(out/"plots/fig_metric_to_auc_pca.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
