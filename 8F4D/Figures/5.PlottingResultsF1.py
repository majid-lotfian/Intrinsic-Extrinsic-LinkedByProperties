#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F1 — Property → AUC trends (SPLIT: one figure per property)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Files are in the same folder as this script
FILES = {
    "PCA":      "results_PCA.csv",
    "TransTab": "results_transtab.csv",
    "TabNet":   "results_tabnet.csv"
}

PROPERTIES = ["eid", "redundancy", "skew", "missing"]
LEVELS     = ["low", "mid", "high"]
AUC_COL    = "auc_logistic_knn"

ENC_STY = {
    "TransTab": {"color": "#1f77b4", "marker": "o"},
    "TabNet":   {"color": "#2ca02c", "marker": "s"},
    "PCA":      {"color": "#7f7f7f", "marker": "^"},
}
ENC_ORDER = ["TransTab", "TabNet", "PCA"]

def load_all():
    dfs = []
    for enc, fname in FILES.items():
        df = pd.read_csv(fname)
        df["encoder"] = enc
        dfs.append(df)
    res = pd.concat(dfs, ignore_index=True)
    res = res[res["property"].isin(PROPERTIES) & res["level"].isin(LEVELS)].copy()
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
    out_fig = Path("figures")
    out_tab = Path("tables")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    res = load_all()

    # Save per-seed points once
    per_seed = res.loc[:, ["file","property","level","seed","encoder",AUC_COL]].copy()
    per_seed.to_csv(out_tab/"F1_points_per_seed.csv", index=False)

    level_idx = {"low":0, "mid":1, "high":2}

    for prop in PROPERTIES:
        sub = res[res["property"]==prop].copy()

        # summary table
        rows = []
        for (enc, lvl), g in sub.groupby(["encoder","level"]):
            m, lo, hi, n = mean_ci(g[AUC_COL])
            rows.append({"property":prop, "level":lvl, "encoder":enc,
                         "mean_auc":m, "ci_low":lo, "ci_high":hi, "n_seeds":n})
        #summ = pd.DataFrame(rows).sort_values(["encoder","level"])

        level_idx = {"low": 0, "mid": 1, "high": 2}
        summ = pd.DataFrame(rows)
        summ["level_num"] = summ["level"].map(level_idx)
        summ = summ.sort_values(["encoder", "level_num"])



        summ.to_csv(out_tab/f"F1_property_{prop}_summary.csv", index=False)

        # figure
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for enc in ENC_ORDER:
            s = summ[summ["encoder"]==enc].copy()
            if s.empty: 
                continue
            s["x"] = s["level"].map(level_idx)
            sty = ENC_STY[enc]
            ax.errorbar(s["level_num"], s["mean_auc"],
                        yerr=[s["mean_auc"]-s["ci_low"], s["ci_high"]-s["mean_auc"]],
                        fmt=sty["marker"]+"-", color=sty["color"], linewidth=2, markersize=6,
                        capsize=3, label=enc, alpha=0.95)


            # Seed points (faint scatter with jitter)
            seeds = sub[sub["encoder"]==enc]
            xs = seeds["level"].map(level_idx).to_numpy(dtype=float)
            xs = xs + (np.random.default_rng(0).normal(0, 0.03, size=len(xs)))  # jitter
            ax.scatter(xs, seeds[AUC_COL], color=sty["color"], alpha=0.25, s=18)

        ax.set_title(f"{prop} → AUC (k-NN, logistic)")
        ax.set_xticks([0,1,2], ["low","mid","high"])
        ax.set_xlabel("level")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncols=3, loc="lower right")

        fig.tight_layout()
        fig.savefig(out_fig/f"F1_property_{prop}_auc.png", dpi=180)
        plt.close(fig)

    print("✅ Wrote figures to ./figures and tables to ./tables")

if __name__ == "__main__":
    main()
