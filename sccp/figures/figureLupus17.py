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
from scipy.stats import linregress, pearsonr, spearmanr


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    cellPercDF = getCellCountPercDF(X, celltype="Cell Type2", cellPerc=True)
    celltype = np.unique(cellPercDF["Cell Type"])
    sns.boxplot(
        data=cellPercDF,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[0],
    )
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)

    cmp = 22
    idx = len(np.unique(cellPercDF["Cell Type"]))
    cellCountDF = getCellCountPercDF(X, celltype="Cell Type2", cellPerc=False)
    plotCmpPerCellCount(X, cmp, cellCountDF, ax[1 : idx + 2], cellPerc=False)

    return f


def getCellCountPercDF(X, celltype="Cell Type", cellPerc=True):
    """Determine cell count or cell type percentage per condition and patient"""
    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2"]].reset_index(
        drop=True
    )
    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby([celltype, "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")
    if cellPerc is True:
        for i, cond in enumerate(np.unique(df["Condition"])):
            dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"] = (
                100
                * dfCellType.loc[
                    dfCellType["Condition"] == cond, "Cell Count"
                ].to_numpy()
                / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
            )
        dfCellType.rename(columns={"Cell Count": "Cell Type Percentage"}, inplace=True)
    dfCellType.rename(
        columns={celltype: "Cell Type", "SLE_status": "Status"}, inplace=True
    )

    return dfCellType


def plotCmpPerCellCount(X, cmp, cellcountDF, ax, cellPerc=True):
    """Plot component weights by cell count for a cell type"""
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
                cellcountDF.loc[cellcountDF["Condition"] == cond]["Status"]
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
                        "Status": status,
                        cellPerc: 0,
                        "Cmp": factorsA[j],
                    }
                )
            totaldf = pd.concat([totaldf, smalldf])
        df = totaldf.loc[totaldf["Cell Type"] == celltype]
        _, _, r_value, _, _ = linregress(df["Cmp"], df[cellPerc])
        pearson = pearsonr(df["Cmp"], df[cellPerc])[0]
        spearman = spearmanr(df["Cmp"], df[cellPerc])[0]

        sns.scatterplot(data=df, x="Cmp", y=cellPerc, hue="Status", ax=ax[i])
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
    ax[-1].set_xticks(ax[-1].get_xticks())
    ax[-1].set_xticklabels(labels=ax[-1].get_xticklabels(), rotation=90)
    ax[-1].set(title=f"Cmp. {cmp} V. Cell Count")
