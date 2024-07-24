"""
Lupus: Average cytotoxic score for each cell type
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
import scanpy as sc
from .commonFuncs.plotGeneral import rotate_xaxis, cell_count_perc_df
from matplotlib.axes import Axes
import anndata
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap
from .figure4f_k import plot_correlation_cmp_cell_count_perc


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    plot_wp_pacmap(X, 4, ax[0], 0.25)
    plot_wp_pacmap(X, 27, ax[1], 0.25)

    X = cytotoxic_score(X)
    plot_score(X, ax[3], cellType="Cell Type2")
    ax[0].set(ylabel="Cytotoxic Score")

    plot_gene_pacmap("RETN", "Pf2", X, ax[4])
    
    celltype_count_perc_df = cell_count_perc_df(X, celltype="leiden", status=True)
    plot_correlation_cmp_cell_count_perc(
            X, 28, celltype_count_perc_df, ax[5], cellPerc=False
        )
    
    

    

    return f


def cytotoxic_score(X: anndata.AnnData):
    """Scanpy average gene score for all cells"""
    cytotoxic_genes = ["PRF1", "GZMH", "GZMB"]
    X = sc.tl.score_genes(adata=X, gene_list=cytotoxic_genes, copy=True, use_raw=False)

    return X


def plot_score(X: anndata, ax: Axes, cellType="Cell Type"):
    """Plots average score  across cell types and patient status"""
    df = pd.DataFrame({"Score": X.obs["score"].values})
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs[cellType].values
    df_mean_score = (
        df.groupby(["Status", "Cell Type", "Condition"], observed=False)
        .mean()
        .reset_index()
        .dropna()
        .sort_values(["Cell Type", "Condition"])
    )

    df_count = (
        df.groupby(["Cell Type", "Condition"], observed=False)
        .size()
        .reset_index(name="Cell Count")
        .sort_values(["Cell Type", "Condition"])
    )

    df_mean_score["Cell Count"] = df_count["Cell Count"].to_numpy()

    sns.boxplot(
        data=df_mean_score,
        x="Cell Type",
        y="Score",
        hue="Status",
        order=np.unique(df["Cell Type"]),
        ax=ax,
        showfliers=False,
    )

    rotate_xaxis(ax)

