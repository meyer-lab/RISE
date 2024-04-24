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
from ..stats import wls_stats_comparison


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
    dfMean = df.groupby(["Status", "Cell Type", "Condition"], observed=False).mean().reset_index(
        ).dropna().sort_values(["Cell Type", "Condition"])
    dfCount = df.groupby(["Cell Type", "Condition"], observed=False).size().reset_index(
        name="Cell Count").sort_values(["Cell Type", "Condition"])

    dfMean["Cell Count"] = dfCount["Cell Count"].to_numpy()
    
    sns.boxplot(
        data=dfMean,
        x="Cell Type",
        y="Score",
        hue="Status",
        order=np.unique(df["Cell Type"]),
        ax=ax,
        showfliers=False)
    
    rotate_xaxis(ax)
    
    pval_df = wls_stats_comparison(dfMean, 
                                  column_comparison_name="Score", 
                                  category_name="Status", 
                                  status_name="SLE")
    
    print(pval_df)
    