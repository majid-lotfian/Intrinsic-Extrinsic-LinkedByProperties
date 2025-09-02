#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F3 — Property → Effective Rank trends (SPLIT: one figure per property)
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
METRIC_COL = "eff_rank"   # <-- ER instead of AUC

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
    per_seed = res.loc[:, ["file","property","level","seed","encoder",METRIC_COL]].copy()
    per_seed.to_csv(out_tab/"F3_points_per_seed.csv", index=False)

    level_idx = {"low":0, "mid":1, "high":2}

    for prop in PROPERTIES:
        sub = res[res["property"]==prop].copy()

        # summary table
        rows = []
        for (enc, lvl), g in sub.groupby(["encoder","level"]):
            m, lo, hi, n = mean_ci(g[METRIC_COL])
            rows.append({"property":prop, "level":lvl, "encoder":enc,
                         "mean_er":m, "ci_low":lo, "ci_high":hi, "n_seeds":n})

        summ = pd.DataFrame(rows)
        summ["level_num"] = summ["level"].map(level_idx)
        summ = summ.sort_values(["encoder", "level_num"])
        summ.to_csv(out_tab/f"F3_property_{prop}_summary.csv", index=False)

        # figure
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for enc in ENC_ORDER:
            s = summ[summ["encoder"]==enc].copy()
            if s.empty: 
                continue
            sty = ENC_STY[enc]
            # line + CI
            ax.errorbar(s["level_num"], s["mean_er"],
                        yerr=[s["mean_er"]-s["ci_low"], s["ci_high"]-s["mean_er"]],
                        fmt=sty["marker"]+"-", color=sty["color"], linewidth=2, markersize=6,
                        capsize=3, label=enc, alpha=0.95)
            # semi-transparent seed points (jittered)
            seeds = sub[sub["encoder"]==enc]
            xs = seeds["level"].map(level_idx).to_numpy(dtype=float)
            xs = xs + (np.random.default_rng(0).normal(0, 0.03, size=len(xs)))
            ax.scatter(xs, seeds[METRIC_COL], color=sty["color"], alpha=0.25, s=18)

        ax.set_title(f"{prop} → Effective Rank")
        ax.set_xticks([0,1,2], ["low","mid","high"])
        ax.set_xlabel("level")
        ax.set_ylabel("Effective Rank")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncols=3, loc="best")

        fig.tight_layout()
        fig.savefig(out_fig/f"F3_property_{prop}_er.png", dpi=180)
        plt.close(fig)

    print("✅ Wrote F3 figures to ./figures and tables to ./tables")

if __name__ == "__main__":
    main()
