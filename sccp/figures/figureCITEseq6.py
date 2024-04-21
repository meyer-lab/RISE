"""
CITEseq: Plotting cell count per Leiden cluster per condition
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
import seaborn as sns
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    plotCellCount(X, ax[0])

    return f


def plotCellCount(X, ax, celltype="leiden"):
    """Plots cell count per cluster per condition and as a percentage"""
    df = X.obs[[celltype, "Condition"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )

    dfCellType = (
        df.groupby([celltype, "Condition"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    dfCellType["Count"] = dfCellType["Count"].astype("float")
    for i, cond in enumerate(pd.unique(df["Condition"])):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
        )
    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
    sns.barplot(
        data=dfCellType,
        x=celltype,
        y="Cell Type Percentage",
        hue="Condition",
        ax=ax,
        errorbar=None,
    )
