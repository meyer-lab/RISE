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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    celltype_count_perc_df = cell_count_perc_df(X, celltype="Cell Type2", status=True)
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="SLE_status",
        order=celltype,
        showfliers=False,
        ax=ax[0],
    )
    rotate_xaxis(ax[0])
    
    pval_df = wls_stats_comparison(celltype_count_perc_df, 
                                  column_comparison_name="Cell Type Percentage", 
                                  category_name="SLE_status", 
                                  status_name="SLE")
    print(pval_df)

    cmp = 22
    idx = len(np.unique(celltype_count_perc_df["Cell Type"]))
    plot_correlation_cmp_cell_count_perc(X, cmp, celltype_count_perc_df, ax[1 : idx + 2], cellPerc=False)

    return f


def plot_correlation_cmp_cell_count_perc(X: anndata, cmp: int, cellcountDF: pd.DataFrame, ax: list[Axes], cellPerc=True):
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
        _, _, r_value, _, _ = linregress(df["Cmp"], df[cellPerc])
        pearson = pearsonr(df["Cmp"], df[cellPerc])[0]
        spearman = spearmanr(df["Cmp"], df[cellPerc])[0]

        sns.scatterplot(data=df, x="Cmp", y=cellPerc, hue="SLE_status", ax=ax[i])
        ax[i].set(
            title=f"{celltype}: R2 Value - {np.round(r_value**2, 3)}",
            xlabel=f"Cmp. {cmp}",
        )

        correl = [np.round(r_value**2, 3), spearman, pearson]
        test = ["R2 Value ", "Pearson", "Spearman"]

        for k in range(3):
            correlationdf = pd.concat(
                [
                    correlationdf,
                    pd.DataFrame(
                        {
                            "Cell Type": celltype,
                            "Correlation": [test[k]],
                            "Value": [correl[k]],
                        }
                    ),
                ]
            )

    sns.barplot(
        data=correlationdf, x="Cell Type", y="Value", hue="Correlation", ax=ax[-1]
    )
    rotate_xaxis(ax[-1])
    ax[-1].set(title=f"Cmp. {cmp} V. {cellPerc}")
