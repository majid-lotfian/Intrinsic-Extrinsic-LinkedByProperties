#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F2 — Property → AUC trends (split by metric)
- Loads results_transtab.csv, results_tabnet.csv, results_PCA.csv
- Reshapes to long format (stack AUC metrics)
- For each property (eid, redundancy, skew, missing):
    * One figure per metric (logistic_lin, rule_lin, logistic_knn, rule_knn)
    * X = {low, mid, high}
    * Y = mean AUC ± 95% CI
    * Trend lines for TransTab, TabNet, PCA
- Outputs:
    figures/F2_<prop>_<metric>.png
    tables/F2_<prop>_<metric>_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_FILES = ["results_transtab8F4D.csv","results_tabnet8F4D.csv","results_PCA8F4D.csv"]
PROPERTIES   = ["eid","redundancy","skew","missing"]
LEVELS       = ["low","mid","high"]

# Color & shape per approach
ENC_STY = {
    "TransTab": {"color":"#1f77b4","marker":"o"},
    "TabNet":   {"color":"#2ca02c","marker":"s"},
    "PCA":      {"color":"#7f7f7f","marker":"^"},
}
ENC_ORDER = ["TransTab","TabNet","PCA"]

# AUC metrics to plot
AUC_METRICS = [
    "auc_logistic_lin",
    "auc_rule_lin",
    "auc_logistic_knn",
    "auc_rule_knn"
]

def load_all():
    dfs=[]
    for fname in RESULT_FILES:
        df = pd.read_csv(fname)
        dfs.append(df)
    res = pd.concat(dfs, ignore_index=True)

    # Canonical encoder names
    def canon(enc: str):
        e = str(enc).lower()
        if "trans" in e:
            return "TransTab"
        if "tabnet" in e:
            return "TabNet"
        return "PCA" if "pca" in e else None

    res["encoder"] = res["encoder"].apply(canon)
    res = res[res["encoder"].notna()].copy()
    return res

def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se
    return m, m - ci, m + ci, n

def main():
    out_fig = Path("figures"); out_fig.mkdir(exist_ok=True)
    out_tab = Path("tables"); out_tab.mkdir(exist_ok=True)

    res = load_all()

    # Reshape to long format: one row per metric
    res_long = res.melt(
        id_vars=["file","property","level","seed","encoder"],
        value_vars=AUC_METRICS,
        var_name="metric",
        value_name="auc"
    )

    level_idx = {"low":0, "mid":1, "high":2}

    for prop in PROPERTIES:
        for metric in AUC_METRICS:
            sub = res_long[(res_long["property"]==prop) & (res_long["metric"]==metric)].copy()
            if sub.empty: 
                continue

            # Summary table
            rows = []
            for (enc,lvl), g in sub.groupby(["encoder","level"]):
                m, lo, hi, n = mean_ci(g["auc"])
                rows.append({"property":prop,"level":lvl,"encoder":enc,
                             "metric":metric,"mean_auc":m,
                             "ci_low":lo,"ci_high":hi,"n_seeds":n})
            summ = pd.DataFrame(rows).sort_values(["encoder","level"])
            summ.to_csv(out_tab/f"F2_{prop}_{metric}_summary.csv", index=False)

            # Figure
            fig, ax = plt.subplots(figsize=(5.2,4.2))
            for enc in ENC_ORDER:
                s = summ[summ["encoder"]==enc].copy()
                if s.empty: continue
                s["x"] = s["level"].map(level_idx)
                s = s.sort_values("x")

                sty = ENC_STY[enc]
                ax.errorbar(s["x"], s["mean_auc"],
                            yerr=[s["mean_auc"]-s["ci_low"], s["ci_high"]-s["mean_auc"]],
                            fmt=sty["marker"]+"-", color=sty["color"],
                            linewidth=2, markersize=6, capsize=3,
                            label=enc, alpha=0.95)

                # per-seed faint points
                seeds = sub[sub["encoder"]==enc]
                xs = seeds["level"].map(level_idx).to_numpy(dtype=float)
                xs = xs + (np.random.default_rng(0).normal(0,0.03,size=len(xs)))
                ax.scatter(xs, seeds["auc"], color=sty["color"], alpha=0.25, s=18)

            ax.set_title(f"{prop} → {metric}")
            ax.set_xticks([0,1,2], ["low","mid","high"])
            ax.set_xlabel("level")
            ax.set_ylabel("AUC")
            ax.set_ylim(0.5,1.02)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, ncols=3, loc="lower right")

            fig.tight_layout()
            fig.savefig(out_fig/f"F2_{prop}_{metric}.png", dpi=180)
            plt.close(fig)

    print("✅ Wrote F2 figures to ./figures and tables to ./tables")

if __name__ == "__main__":
    main()
