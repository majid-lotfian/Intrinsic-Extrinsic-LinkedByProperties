#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransTab on synthetic CBC (no embedding files):
- Reads ./synthetic_cbc.zip
- Rebuilds 80/20 split from filename seed
- Self-supervised pretrain (contrastive) on train
- Build encoder from ckpt; encode train/test (in RAM only)
- Metrics: Trustworthiness, Continuity, Effective Rank, Hubness
- Utility: AUC (kNN & Logistic) with the same synthetic labels as PCA pipeline
- Writes: outputs/results_transtab.csv, per-encoder plots/tables
"""

import os, re, zipfile, warnings, tempfile, shutil, gc
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
DATA_ARCHIVE = "synthetic_cbc.zip"     # or ./synthetic_cbc/ folder
OUTPUT_DIR   = "TransTabOutputs"
FEATURES     = ["RBC","HGB","HCT","MCV","MCH","RDW","WBC","PLT"]
TEST_SIZE    = 0.20
HIDDEN_DIM   = 128           # TransTab encoder dim (CLS)
EPOCHS       = 12            # keep light for laptop
BATCH_SIZE   = 128
PATIENCE     = 3
PCA_D4_FAIR  = True          # also score a PCA→4 projection (apples-to-apples vs PCA)
PROP_MARKER  = {"eid":"o","redundancy":"s","skew":"^","missing":"D"}
FILENAME_RE  = re.compile(r"syn_(?P<prop>eid|redundancy|skew|missing)-(?P<level>low|mid|high)_seed(?P<seed>\d+)\.csv")

# shared deterministic logistic label vector (same as before)
_RNG_LABEL = np.random.default_rng(999)
W_GLOBAL = None

# ---------- TransTab imports (lazy) ----------
import transtab  # official library: build_contrastive_learner/train/build_encoder

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

# ---------- main ----------
def main():
    out = Path(OUTPUT_DIR)
    (out/"plots").mkdir(parents=True, exist_ok=True)
    (out/"tables").mkdir(parents=True, exist_ok=True)

    kind, container, names = list_inputs()
    rows = []

    for name in tqdm(names, desc="TransTab"):
        meta = parse_meta(name)
        seed = meta["seed"]

        # load & split
        df = read_csv(kind, container, name)
        if "id" not in df.columns:
            df.insert(0, "id", np.arange(1, len(df)+1, dtype=np.int64))
        X = df[FEATURES].copy()
        ids = df["id"].to_numpy()
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, shuffle=True)
        # small val from train for early stopping
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=seed, shuffle=True)

        # impute for metrics/labels (same protocol as PCA)
        imp = SimpleImputer(strategy="mean")
        Xtr = imp.fit_transform(X.iloc[tr_idx2]); Xva = imp.transform(X.iloc[va_idx]); Xte = imp.transform(X.iloc[te_idx])
        scaler = StandardScaler()
        Xtrz = scaler.fit_transform(Xtr); Xtez = scaler.transform(Xte)  # z-scored only used for labels + trust/cont space
        # labels (unchanged from PCA pipeline)
        y_log_all  = make_logistic_label_global(scaler.transform(imp.transform(X)))
        y_rule_all = make_rule_label(pd.DataFrame(imp.transform(X), columns=FEATURES))
        y_log_tr,  y_log_te  = y_log_all[tr_idx2], y_log_all[te_idx]
        y_rule_tr, y_rule_te = y_rule_all[tr_idx2], y_rule_all[te_idx]

        # ---- TransTab: contrastive pretrain, then encoder ----
        # Build contrastive learner
        model, collate_fn = transtab.build_contrastive_learner(
            categorical_columns=[], numerical_columns=FEATURES, binary_columns=[]
        )  # returns (model, collate_fn)
        # Train (self-supervised)
        trainset = (X.iloc[tr_idx2], pd.Series(np.zeros(len(tr_idx2), dtype=int)))
        valset   = (X.iloc[va_idx],  pd.Series(np.zeros(len(va_idx),  dtype=int)))
        with tempfile.TemporaryDirectory() as ckpt_dir:
            transtab.train(model, trainset, valset,
                           num_epoch=EPOCHS, batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE,
                           lr=1e-4, patience=PATIENCE, eval_metric='val_loss', eval_less_is_better=True,
                           output_dir=ckpt_dir, collate_fn=collate_fn, load_best_at_last=True)
            # Build encoder from checkpoint and encode (CLS)
            enc = transtab.build_encoder(
                categorical_columns=[], numerical_columns=FEATURES, binary_columns=[],
                hidden_dim=HIDDEN_DIM, checkpoint=ckpt_dir
            )
            Ztr = enc(X.iloc[tr_idx2]).detach().cpu().numpy()
            Zte = enc(X.iloc[te_idx]).detach().cpu().numpy()

        # variants: native + PCA→4 (optional)
        variants = [("TransTab_native", Ztr, Zte)]
        if PCA_D4_FAIR and Ztr.shape[1] != 4:
            p = PCA(n_components=4, random_state=0).fit(Ztr)
            variants.append(("TransTab_d4", p.transform(Ztr), p.transform(Zte)))

        # metrics + utility (on test)
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

        # clean memory
        del model, enc, Ztr, Zte, Xtr, Xva, Xte, Xtrz, Xtez
        gc.collect()

    # ----- save results + quick plots -----
    out = Path(OUTPUT_DIR)
    dfres = pd.DataFrame(rows)
    dfres.to_csv(out/"results_transtab8F4D.csv", index=False)

    # per-encoder plot ER→AUC
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

    # exact points for figureB (like before)
    tables = out/"tables"; tables.mkdir(parents=True, exist_ok=True)
    dfres.loc[:, ["file","property","level","seed","encoder","eff_rank","auc_logistic_knn"]].to_csv(
        tables/"transtab_figureB_points.csv", index=False
    )
    print(f"[OK] Wrote {out/'results_transtab.csv'} with {len(dfres)} rows.")

if __name__ == "__main__":
    main()
