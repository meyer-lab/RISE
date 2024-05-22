"""
Lupus:
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .commonFuncs.plotGeneral import plot_avegene_per_status, rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    genes = ["IFITM3", "APOBEC3A"]

    df_total = pd.DataFrame([])

    for i, gene in enumerate(np.ravel(genes)):
        df = plot_avegene_per_status(X, gene, ax[i], cellType="Cell Type2")
        rotate_xaxis(ax[i])
        df_total = pd.concat([df, df_total])

    plot_ave2genes_per_status(df_total, genes[0], genes[1], ax[2])

    return f


def plot_ave2genes_per_status(df_total, gene1, gene2, ax):
    """Plots average of 2 genes per celltype per status"""
    df_total = df_total.pivot(
        index=["Status", "Cell Type", "Condition"],
        columns="Gene",
        values="Average Gene Expression",
    )
    df_mean = (
        df_total.groupby(["Status", "Cell Type"], observed=False)
        .mean()
        .dropna()
        .reset_index()
    )
    df_std = (
        df_total.groupby(["Status", "Cell Type"], observed=False)
        .std()
        .dropna()
        .reset_index()
    )

    colors = sns.color_palette("hls", len(np.unique(df_mean["Cell Type"])))
    fmt = ["o", "*"]

    for i, status in enumerate(np.unique(df_mean["Status"])):
        for j, celltype in enumerate(np.unique(df_mean["Cell Type"])):
            df_mini_mean = df_mean.loc[
                (df_mean["Status"] == status) & (df_mean["Cell Type"] == celltype)
            ]
            df_mini_std = df_std.loc[
                (df_std["Status"] == status) & (df_std["Cell Type"] == celltype)
            ]
            ax.errorbar(
                df_mini_mean[gene1],
                df_mini_mean[gene2],
                xerr=df_mini_std[gene1],
                yerr=df_mini_std[gene2],
                ls="none",
                fmt=fmt[i],
                label=celltype + status,
                color=colors[j],
            )

    ax.set(xlabel=f"Average {gene1}", ylabel=f"Average {gene2}")
    ax.legend()
