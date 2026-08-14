"""
Generate two slide figures:
  1. Strategy-1 rho bar chart: run_final (20 samples/variant) vs run_clean (50 samples/variant)
  2. Cross-dataset transfer heatmap (run_final, raw_gnn)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

OUT = Path("output/run_clean_20260730/analysis/slides")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["History", "Amazon", "ArXiv", "Electronics", "Toys"]

# ── 1. Strategy-1 ρ bar chart ─────────────────────────────────────────────
# Old: run_final / raw_gnn diagonal (20 samples per variant)
old_rho = {
    "History":     0.7798,
    "Amazon":      0.9459,
    "ArXiv":       0.9625,
    "Electronics": 0.9279,
    "Toys":        0.5825,
}
# New: run_clean_20260730 / experiment1 / raw_gnn (50 samples per variant)
new_rho = {
    "History":     0.8839,
    "Amazon":      0.8903,
    "ArXiv":       0.9744,
    "Electronics": 0.9152,
    "Toys":        0.6625,
}

x = np.arange(len(DATASETS))
w = 0.35
old_vals = [old_rho[d] for d in DATASETS]
new_vals = [new_rho[d] for d in DATASETS]

fig, ax = plt.subplots(figsize=(9, 5))
bars_old = ax.bar(x - w/2, old_vals, width=w, label="run_final (20 samples/variant)",
                  color="#6baed6", edgecolor="white", linewidth=0.5)
bars_new = ax.bar(x + w/2, new_vals, width=w, label="run_clean (50 samples/variant)",
                  color="#2171b5", edgecolor="white", linewidth=0.5)

# Annotate with ρ value and Δρ
for i, (ov, nv) in enumerate(zip(old_vals, new_vals)):
    delta = nv - ov
    sign = "+" if delta >= 0 else ""
    ax.text(x[i] - w/2, ov + 0.012, f"{ov:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.text(x[i] + w/2, nv + 0.012, f"{nv:.3f}", ha="center", va="bottom", fontsize=8,
            color="#003f8a", fontweight="bold")
    col = "#1a7a1a" if delta >= 0 else "#c0392b"
    ax.text(x[i] + w/2, nv + 0.055, f"({sign}{delta:.3f})", ha="center", va="bottom",
            fontsize=7.5, color=col)

ax.set_xticks(x)
ax.set_xticklabels(DATASETS, fontsize=12)
ax.set_ylabel("Spearman ρ (Strategy-1)", fontsize=11)
ax.set_title("Strategy-1 ρ: Prior Run vs Clean Run (raw_gnn, no pruning)", fontsize=12, pad=10)
ax.set_ylim(0, 1.12)
ax.axhline(0.9, color="gray", lw=0.8, ls="--", alpha=0.5, label="ρ = 0.90")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "strategy1_rho_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'strategy1_rho_comparison.png'}")


# ── 2. Cross-dataset transfer heatmap ─────────────────────────────────────
# Table B = ArXiv M2/M6 excluded symmetrically (clean), from run_clean_20260730
rho_df = pd.read_csv("output/analysis/cross_dataset_3x3_clean_rho.csv", index_col=0)
p_df   = pd.read_csv("output/analysis/cross_dataset_3x3_clean_p.csv",   index_col=0)

# Short names matching index/columns
short = {"history": "Hist", "amazon": "Amzn", "arxiv": "ArXv", "electronics": "Elec", "toys": "Toys"}
rho_df.index   = [short[i] for i in rho_df.index]
rho_df.columns = [short[c] for c in rho_df.columns]
p_df.index     = [short[i] for i in p_df.index]
p_df.columns   = [short[c] for c in p_df.columns]

mat   = rho_df.values.astype(float)
p_mat = p_df.values.astype(float)
n     = len(rho_df)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
cmap = plt.cm.RdBu
norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

# Cell annotations
for i in range(n):
    for j in range(n):
        rho_val = mat[i, j]
        p_val   = p_mat[i, j]
        stars   = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        diag    = (i == j)
        fc      = "white" if abs(rho_val) > 0.5 else "black"
        rho_str = f"{rho_val:+.3f}"
        label   = f"{'←' if diag else ''}{rho_str}{stars}"
        ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                color=fc, fontweight="bold" if diag else "normal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(rho_df.columns, fontsize=10)
ax.set_yticklabels(rho_df.index,   fontsize=10)
ax.set_xlabel("Eval dataset (test)", fontsize=10, labelpad=6)
ax.set_ylabel("Train dataset (DT fit on)", fontsize=10, labelpad=6)
ax.set_title("Cross-dataset Strategy-1 ρ transfer (raw_gnn)", fontsize=11, pad=8)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Spearman ρ", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Diagonal box
for i in range(n):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="gold", lw=2.5))

fig.tight_layout()
fig.savefig(OUT / "cross_dataset_transfer_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'cross_dataset_transfer_heatmap.png'}")

print("\nDone.")
