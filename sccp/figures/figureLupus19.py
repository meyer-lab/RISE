"""
Lupus
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
from ..figures.commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((4, 2), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    X = cytotoxic_score(X)
    plot_score(X, ax[0], cellType="Cell Type2")
    ax[0].set(ylabel="Cytotoxic Score")
    
    
    return f


def cytotoxic_score(X):
    """Scanpy average gene score for all cells"""
    cytotoxic_genes = ["PRF1", "GZMH", "GZMB"]
    X = sc.tl.score_genes(adata=X, gene_list=cytotoxic_genes, copy=True, use_raw=False)

    return X


def plot_score(X, ax, cellType="Cell Type"):
    """Plots average score  across cell types and patient status"""
    df = pd.DataFrame({"Score": X.obs["score"].values})
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs[cellType].values
    df = df.groupby(["Status", "Cell Type", "Condition"], observed=False).mean().reset_index()
    
    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Score",
        hue="Status",
        order=np.unique(df["Cell Type"]),
        ax=ax,
        showfliers=False)
    
    rotate_xaxis(ax)