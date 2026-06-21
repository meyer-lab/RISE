"""Generate figures for the RISE tutorial documentation."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt

# Create output directory
output_dir = Path(__file__).parent / "_static" / "tutorial_images"
output_dir.mkdir(parents=True, exist_ok=True)

# Import RISE modules
from analysis.imports import import_thomson
from RISE.factorization import pf2
from RISE.plotting import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_fms_diff_ranks,
    plot_gene_factors,
    plot_gene_pacmap,
    plot_labels_pacmap,
    plot_r2x,
    plot_wp_pacmap,
)

print("Loading dataset...")
X = import_thomson()

# Figure 1: Variance Explained (R2X)
print("Generating Figure 1: R2X plot...")
ranks = [1, 5, 10, 15, 20, 25, 30]
fig, ax = plt.subplots(figsize=(5, 5))
plot_r2x(X, ranks, ax)
plt.tight_layout()
plt.savefig(output_dir / "step2_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Factor Match Score
print("Generating Figure 2: FMS plot...")
fig, ax = plt.subplots(figsize=(5, 5))
plot_fms_diff_ranks(X, ax, ranksList=list(ranks), runs=3)
plt.tight_layout()
plt.savefig(output_dir / "step3_fms.png", dpi=150, bbox_inches="tight")
plt.close()

# Perform RISE factorization
print("Running RISE factorization...")
rank = 20
X = pf2(X=X, rank=rank, doEmbedding=True, tolerance=1e-9, max_iter=500, random_state=42)

# Figure 3: Condition Factors
print("Generating Figure 3: Condition factors...")
fig, ax = plt.subplots(figsize=(8, 8))
plot_condition_factors(X, ax=ax, cond="Condition", log_transform=True)
plt.tight_layout()
plt.savefig(output_dir / "step5_condition_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Cell Embedding
print("Generating Figure 4: Cell embedding...")
fig, ax = plt.subplots(figsize=(8, 8))
plot_labels_pacmap(X, labelType="Cell Type", ax=ax)
plt.tight_layout()
plt.savefig(output_dir / "step6_cell_embedding.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 5: Eigen-state Factors
print("Generating Figure 5: Eigen-state factors...")
fig, ax = plt.subplots(figsize=(4, 4))
plot_eigenstate_factors(X, ax=ax)
plt.ylabel("Eigen-state")
plt.tight_layout()
plt.savefig(output_dir / "step7_eigenstate_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 6: Gene Factors
print("Generating Figure 6: Gene factors...")
fig, ax = plt.subplots(figsize=(7, 8))
plot_gene_factors(X, ax=ax, weight=0.2, trim=True)
plt.tight_layout()
plt.savefig(output_dir / "step8_gene_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 7: Gene Expression on PaCMAP
print("Generating Figure 7: Gene expression...")
fig, ax = plt.subplots(figsize=(8, 8))
gene = "MS4A1"
plot_gene_pacmap(gene, X, ax=ax, clip_outliers=0.9995)
plt.tight_layout()
plt.savefig(output_dir / "step9_gene_expression.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 8: Weighted Projections
print("Generating Figure 8: Weighted projections...")
fig, ax = plt.subplots(figsize=(8, 8))
plot_wp_pacmap(X, cmp=10, ax=ax, cbarMax=0.9)
plt.tight_layout()
plt.savefig(
    output_dir / "step10_weighted_projections.png", dpi=150, bbox_inches="tight"
)
plt.close()

print(f"\nAll figures saved to {output_dir}")
print("Done!")
