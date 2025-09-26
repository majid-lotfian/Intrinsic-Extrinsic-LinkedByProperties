#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransTab on real datasets (4D, 16D, 32D):
- Reads 4 tables placed next to this script: CBC.csv, Covid19.csv, Iraq.xlsx, Liverpool.csv
- Universal loader (CSV/XLSX), header cleaning, sex→{0,1}, drop dates/non-numerics, keep age
- 80/20 split (seed=42); from train, hold 10% as validation for early stopping
- Self-supervised TransTab pretraining per embedding dimension in {4, 16, 32}
- Encode train/test into embeddings (CLS)
- Intrinsic metrics on test: Trustworthiness, Continuity, Effective Rank, Hubness
- Utility metrics: AUC (LogReg + kNN) for logistic label; rule-based AUC if WBC/PLT/HGB available
- Property stats on raw features + synthetic-metadata-driven low/mid/high levels (EID/Redundancy/Skew/Missingness)

Outputs under TransTabReal/:
  results_transtab.csv                     # one row per dataset × hidden_dim
  embeddings/<dataset>_<dim>d_train.npz
  embeddings/<dataset>_<dim>d_test.npz
  models/<dataset>_<dim>d/...
  tables/real_property_stats.csv
  tables/real_property_levels.csv
  tables/thresholds_used.csv
  plots/er_to_auc_4d.png
  plots/er_to_auc_16d.png
  plots/er_to_auc_32d.png
"""

import warnings, gc
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

# TransTab library (pip install transtab)
import transtab  # provides build_contrastive_learner, train, build_encoder

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
OUTPUT_DIR = "TransTabReal"
DATASETS = ["CBC.csv", "Covid19.csv", "Iraq.xlsx", "Liverpool.csv"]
HIDDEN_DIMS = [4, 16, 32]  # embeddings to produce

# choose attention heads that divide the hidden dim
def pick_heads(dim: int) -> int:
    for h in (8, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1  # fallback, should never hit

# ---------- Header cleaning & IO ----------

def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        # needs openpyxl
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Clean column names: strip, unquote, spaces/tabs -> underscore
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

    # keep ids for reference (but drop from features)
    id_col = None
    for c in df.columns:
        if c.lower() == "id":
            id_col = c
            break
    ids = df[id_col].astype("Int64") if id_col else pd.Series(np.arange(1, len(df)+1, dtype=np.int64), name="id")

    # encode sex/gender
    for c in df.columns:
        if c.lower() in ["sex", "gender"]:
            df[c] = encode_sex(df[c])

    # drop date-like columns
    drop_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    # numeric-only features; drop id if present
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
    ind_X = nbr_X.kneighbors(return_distance=False)[:, 1:]
    nbr_Z = NearestNeighbors(n_neighbors=n-1).fit(Z)
    ind_Z = nbr_Z.kneighbors(return_distance=False)
    ranks = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        ranks[i, ind_Z[i, :]] = np.arange(1, n)
        ranks[i, i] = 0
    pen = 0.0
    for i in range(n):
        neighZk = set(ind_Z[i, :k])
        for j in ind_X[i]:
            if j not in neighZk:
                r = ranks[i, j]
                if r > 0: pen += (r - k)
    denom = (2.0 / (n * k * (2 * n - 3 * k - 1))) if (2 * n - 3 * k - 1) > 0 else 0.0
    return float(1.0 - denom * pen)

def effective_rank(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    S = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    lam = (S**2).astype(np.float64)
    if lam.sum() <= 0: return float("nan")
    p = lam / lam.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))

def hubness_skew(Z, k):
    n = Z.shape[0]; k = min(k, n-1)
    inds = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    c = np.bincount(inds.ravel(), minlength=n)
    return 0.0 if c.std() == 0 else float(stats.skew(c))

# ---------- Logistic label (dimension-aware cache) ----------

_W_CACHE: dict[int, np.ndarray] = {}

def make_logistic_label_global(Xz: np.ndarray) -> np.ndarray:
    """
    Deterministic synthetic label per input dimension d = Xz.shape[1].
    Balanced via median threshold per dataset.
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
    if X_complete.shape[0] >= max(30, d + 5):
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
    # EID normalized by synth d
    sdf = meta[meta["property"] == "eid"].copy()
    t1, t2 = midpoints(
        sdf[sdf["level"] == "low"]["realized_eff_rank"] / d_synth,
        sdf[sdf["level"] == "mid"]["realized_eff_rank"] / d_synth,
        sdf[sdf["level"] == "high"]["realized_eff_rank"] / d_synth,
    )
    cuts["eid"] = (t1, t2)
    # Redundancy
    sdf = meta[meta["property"] == "redundancy"]
    t1, t2 = midpoints(
        sdf[sdf["level"] == "low"]["realized_avg_abs_corr"],
        sdf[sdf["level"] == "mid"]["realized_avg_abs_corr"],
        sdf[sdf["level"] == "high"]["realized_avg_abs_corr"],
    )
    cuts["redundancy"] = (t1, t2)
    # Skew
    sdf = meta[meta["property"] == "skew"]
    t1, t2 = midpoints(
        sdf[sdf["level"] == "low"]["realized_mean_abs_skew"],
        sdf[sdf["level"] == "mid"]["realized_mean_abs_skew"],
        sdf[sdf["level"] == "high"]["realized_mean_abs_skew"],
    )
    cuts["skew"] = (t1, t2)
    # Missingness
    sdf = meta[meta["property"] == "missing"]
    t1, t2 = midpoints(
        sdf[sdf["level"] == "low"]["realized_missing_rate"],
        sdf[sdf["level"] == "mid"]["realized_missing_rate"],
        sdf[sdf["level"] == "high"]["realized_missing_rate"],
    )
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

# ---------- Main pipeline ----------

def main():
    here = Path(__file__).parent
    out = here / OUTPUT_DIR
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # thresholds from synthetic (24F) metadata
    meta_path = here / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.csv (24-feature synthetic metadata) not found next to this script.")
    cuts = load_thresholds_from_metadata(meta_path, d_synth=24)
    pd.DataFrame([
        {"property": "eid", "cut_low_mid": cuts["eid"][0], "cut_mid_high": cuts["eid"][1], "normalized": "ER_norm"},
        {"property": "redundancy", "cut_low_mid": cuts["redundancy"][0], "cut_mid_high": cuts["redundancy"][1]},
        {"property": "skew", "cut_low_mid": cuts["skew"][0], "cut_mid_high": cuts["skew"][1]},
        {"property": "missing", "cut_low_mid": cuts["missing"][0], "cut_mid_high": cuts["missing"][1]},
    ]).to_csv(out / "tables/thresholds_used.csv", index=False)

    all_rows = []
    prop_rows = []
    # compute property stats ONCE per dataset (raw features)
    per_dataset_stats = {}

    for fname in DATASETS:
        path = here / fname
        if not path.exists():
            print(f"[WARN] Missing dataset: {fname}; skipping.")
            continue

        print(f"[INFO] Loading {fname}")
        df_raw, df_num, ids_all = load_dataset(path)
        stem = path.stem

        # property stats/levels on raw numeric features
        stats_dict = dataset_property_stats(df_num)
        levels = classify_levels(stats_dict, cuts)
        per_dataset_stats[stem] = (stats_dict, levels)

        # rule label mapping on raw df
        rule_map = normalize_rule_columns(df_raw)

        # common split (shared across dims)
        idx = np.arange(len(df_num))
        tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.10, random_state=SEED, shuffle=True)

        # impute/scale on FULL TRAIN (for labels & trust/cont)
        imp = SimpleImputer(strategy="mean")
        Xtr  = imp.fit_transform(df_num.iloc[tr_idx])
        Xva  = imp.transform(df_num.iloc[va_idx])
        Xte  = imp.transform(df_num.iloc[te_idx])
        Xall = imp.transform(df_num)

        scaler = StandardScaler()
        Xtrz = scaler.fit_transform(Xtr)
        Xtez = scaler.transform(Xte)
        Xallz = scaler.transform(Xall)

        # labels
        y_log_all = make_logistic_label_global(Xallz)
        y_log_tr2, y_log_te = y_log_all[tr_idx2], y_log_all[te_idx]

        y_rule_all = make_rule_label(df_raw, rule_map)
        if y_rule_all is not None:
            y_rule_tr2, y_rule_te = y_rule_all[tr_idx2], y_rule_all[te_idx]
        else:
            y_rule_tr2 = y_rule_te = None

        n_test = Xtez.shape[0]
        k1 = max(5, int(np.floor(np.sqrt(n_test))))
        k2 = min(n_test-1, max(10, 2*k1))

        # TransTab per dimension
        numeric_cols = list(df_num.columns)
        for dim in HIDDEN_DIMS:
            print(f"[INFO] {stem}: training TransTab (hidden_dim={dim})")
            # build learner
            nheads = pick_heads(dim)  # 4D->1 head, 16D->4, 32D->8

            model, collate_fn = transtab.build_contrastive_learner(
                categorical_columns=[],
                numerical_columns=numeric_cols,
                binary_columns=[],
                hidden_dim=dim,
                num_attention_head=nheads,   
                device="cuda"                
            )

            # train self-supervised
            trainset = (df_num.iloc[tr_idx2], pd.Series(np.zeros(len(tr_idx2), dtype=int)))
            valset   = (df_num.iloc[va_idx],  pd.Series(np.zeros(len(va_idx),  dtype=int)))

            ckpt_dir = out / "models" / f"{stem}_{dim}d"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            transtab.train(
                model, trainset, valset,
                num_epoch=30, batch_size=256, eval_batch_size=256,
                lr=1e-4, patience=5, eval_metric="val_loss", eval_less_is_better=True,
                output_dir=str(ckpt_dir),
                collate_fn=collate_fn,
                load_best_at_last=True,
                seed=SEED
            )

            # build encoder from best checkpoint & encode (CLS)
            enc = transtab.build_encoder(
                categorical_columns=[],
                numerical_columns=numeric_cols,
                binary_columns=[],
                hidden_dim=dim,
                num_attention_head=nheads,  
                checkpoint=str(ckpt_dir),
                device="cuda"
            )

            Ztr = enc(df_num.iloc[tr_idx2]).detach().cpu().numpy()
            Zte = enc(df_num.iloc[te_idx]).detach().cpu().numpy()

            # save embeddings
            emb_tr = out / "embeddings" / f"{stem}_{dim}d_train.npz"
            emb_te = out / "embeddings" / f"{stem}_{dim}d_test.npz"
            np.savez_compressed(emb_tr, id=ids_all.iloc[tr_idx2].to_numpy(dtype=np.int64), Z=Ztr.astype(np.float32))
            np.savez_compressed(emb_te, id=ids_all.iloc[te_idx].to_numpy(dtype=np.int64), Z=Zte.astype(np.float32))

            # intrinsic metrics (test)
            trust_k1 = skl_trustworthiness(Xtez, Zte, n_neighbors=k1)
            trust_k2 = skl_trustworthiness(Xtez, Zte, n_neighbors=k2)
            cont_k1  = continuity(Xtez, Zte, n_neighbors=k1)
            cont_k2  = continuity(Xtez, Zte, n_neighbors=k2)
            effr     = effective_rank(Zte)
            hub_k1   = hubness_skew(Zte, k1)
            hub_k2   = hubness_skew(Zte, k2)

            # utility AUCs (test)
            def auc_lin(Ztr_, ytr_, Zte_, yte_):
                if len(np.unique(ytr_)) < 2 or len(np.unique(yte_)) < 2: return np.nan
                clf = LogisticRegression(max_iter=300, solver="lbfgs")
                clf.fit(Ztr_, ytr_); p = clf.predict_proba(Zte_)[:, 1]
                return roc_auc_score(yte_, p)

            def auc_knn(Ztr_, ytr_, Zte_, yte_, k):
                if len(np.unique(ytr_)) < 2 or len(np.unique(yte_)) < 2: return np.nan
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(Ztr_, ytr_); p = knn.predict_proba(Zte_)[:, 1]
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
                "encoder": "TransTab",
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

            # cleanup graph memory
            del model, enc, Ztr, Zte
            gc.collect()

        # after all dims, record property stats once
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

    # save results
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out / "results_transtab.csv", index=False)
    prop_df = pd.DataFrame(prop_rows)
    prop_df.to_csv(out / "tables" / "real_property_stats.csv", index=False)
    prop_df.to_csv(out / "tables" / "real_property_levels.csv", index=False)

    # plots: ER → AUC (logistic kNN), per dim
    if not out_df.empty:
        for dim in sorted(out_df["hidden_dim"].unique()):
            sub = out_df[out_df["hidden_dim"] == dim]
            plt.figure(figsize=(8, 5))
            for ds, grp in sub.groupby("dataset"):
                plt.scatter(grp["eff_rank"], grp["auc_logistic_knn"], label=ds, s=48, alpha=0.9)
            plt.xlabel("Effective Rank (embedding)")
            plt.ylabel("AUC (k-NN, logistic label)")
            plt.title(f"Intrinsic (Effective Rank) → Utility (k-NN AUC) — TransTab {dim}D")
            plt.legend(title="Dataset")
            plt.tight_layout()
            plt.savefig(out / "plots" / f"er_to_auc_{dim}d.png", dpi=150)
            plt.close()

    print(f"[OK] Wrote {out/'results_transtab.csv'} with {len(out_df)} rows.")

if __name__ == "__main__":
    main()
