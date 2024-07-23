"""
Lupus: Cell type percentage between status (with stats comparison) and
correlation between component and cell count/percentage for each cell type
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from ..stats import wls_stats_comparison
from matplotlib.axes import Axes
import anndata
from .commonFuncs.plotGeneral import plot_avegene_per_status


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    celltype_count_perc_df = cell_count_perc_df(X, celltype="Cell Type2", status=True)

    cmps = [14, 22]

    for i, cmp in enumerate(cmps):
        plot_correlation_cmp_cell_count_perc(
            X, cmp, celltype_count_perc_df, ax[i], cellPerc=False
        )

    genes = ["IFITM3", "APOBEC3A"]

    df_total = pd.DataFrame([])

    for i, gene in enumerate(np.ravel(genes)):
        df = plot_avegene_per_status(X, gene, ax[i], cellType="Cell Type2")
        rotate_xaxis(ax[i])
        df_total = pd.concat([df, df_total])

    plot_ave2genes_per_status(df_total, genes[0], genes[1], ax[2])

    return f


def plot_correlation_cmp_cell_count_perc(
    X: anndata, cmp: int, cellcountDF: pd.DataFrame, ax: Axes, cellPerc=True
):
    """Plot component weights by cell type count or percentage for a cell type"""
    yt = np.unique(X.obs["Condition"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, cmp - 1]
    if cellPerc is True:
        cellPerc = "Cell Type Percentage"
    else:
        cellPerc = "Cell Count"
    totaldf = pd.DataFrame([])
    correlationdf = pd.DataFrame([])
    cellcountDF["Condition"] = pd.Categorical(cellcountDF["Condition"], yt)
    for i, celltype in enumerate(np.unique(cellcountDF["Cell Type"])):
        for j, cond in enumerate(np.unique(cellcountDF["Condition"])):
            status = np.unique(
                cellcountDF.loc[cellcountDF["Condition"] == cond]["SLE_status"]
            )
            smalldf = cellcountDF.loc[
                (cellcountDF["Condition"] == cond)
                & (cellcountDF["Cell Type"] == celltype)
            ]
            if smalldf.empty is False:
                smalldf = smalldf.assign(Cmp=factorsA[j])
            else:
                smalldf = pd.DataFrame(
                    {
                        "Condition": cond,
                        "Cell Type": celltype,
                        "SLE_status": status,
                        cellPerc: 0,
                        "Cmp": factorsA[j],
                    }
                )

            totaldf = pd.concat([totaldf, smalldf])

        df = totaldf.loc[totaldf["Cell Type"] == celltype]
        pearson = pearsonr(df["Cmp"], df[cellPerc])[0]

        correlationdf = pd.DataFrame(
            {
                "Cell Type": celltype,
                "Correlation": ["Pearson"],
                "Value": [pearson],
            }
        )
    sns.swarmplot(
        data=correlationdf, x="Cell Type", y="Value", hue="Correlation", ax=ax
    )
    rotate_xaxis(ax)
    ax.set(title=f"Cmp. {cmp} V. {cellPerc}")


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
