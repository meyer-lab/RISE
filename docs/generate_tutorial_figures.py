"""Generate figures for the RISE tutorial documentation dynamically during doc build."""

import os
from pathlib import Path

import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from analysis.imports import import_thomson_factors
from RISE.plotting import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
    plot_gene_pacmap,
    plot_labels_pacmap,
    plot_wp_pacmap,
)


def generate_figures() -> None:
    """Generate dynamic tutorial figures (steps 5-10)."""
    output_dir = Path(__file__).parent / "assets" / "tutorial_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Thomson dataset factors for tutorial figure generation...")
    X = import_thomson_factors(include_raw=True)

    # Figure 3: Condition Factors
    print("Generating Figure 3: Condition factors...")
    _, ax = plt.subplots(figsize=(8, 8))
    plot_condition_factors(X, ax=ax, cond="Condition", log_transform=True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "step5_condition_factors.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Figure 4: Cell Embedding
    print("Generating Figure 4: Cell embedding...")
    _, ax = plt.subplots(figsize=(8, 8))
    plot_labels_pacmap(X, labelType="Cell Type", ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "step6_cell_embedding.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 5: Eigen-state Factors
    print("Generating Figure 5: Eigen-state factors...")
    _, ax = plt.subplots(figsize=(4, 4))
    plot_eigenstate_factors(X, ax=ax)
    plt.ylabel("Eigen-state")
    plt.tight_layout()
    plt.savefig(
        output_dir / "step7_eigenstate_factors.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Figure 6: Gene Factors
    print("Generating Figure 6: Gene factors...")
    _, ax = plt.subplots(figsize=(7, 8))
    plot_gene_factors(X, ax=ax, weight=0.2, trim=True)
    plt.tight_layout()
    plt.savefig(output_dir / "step8_gene_factors.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 7: Gene Expression on PaCMAP
    print("Generating Figure 7: Gene expression...")
    _, ax = plt.subplots(figsize=(8, 8))
    gene = "MS4A1"
    plot_gene_pacmap(gene, X, ax=ax, clip_outliers=0.9995)
    plt.tight_layout()
    plt.savefig(output_dir / "step9_gene_expression.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 8: Weighted Projections
    print("Generating Figure 8: Weighted projections...")
    _, ax = plt.subplots(figsize=(8, 8))
    plot_wp_pacmap(X, cmp=10, ax=ax, cbarMax=0.9)
    plt.tight_layout()
    plt.savefig(
        output_dir / "step10_weighted_projections.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"Tutorial figures successfully generated in {output_dir}")


def on_pre_build(config=None, **kwargs) -> None:
    """MkDocs hook executed before building the documentation."""
    # Ensure ANNDATA_CUPY is set to 0 for doc build
    os.environ["ANNDATA_CUPY"] = "0"
    generate_figures()


if __name__ == "__main__":
    generate_figures()
