"""
Investigation of raw data for Thomson dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, openPf2
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (6, 6))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    rank = 30
    X = openPf2(rank, "Thomson")

    dataDF = pd.DataFrame(
        {"Cell Type": gateThomsonCells(X), "Condition": X.obs["Condition"]}
    )

    dfCond = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Number")
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")

    dfCellType = (
        dataDF.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count")
    )
    for i, cond in enumerate(pd.unique(dataDF["Condition"])):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
        )

    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
    for i, celltype in enumerate(np.unique(dfCellType["Cell Type"])):
        sns.histplot(
            data=dfCellType.loc[dfCellType["Cell Type"] == celltype],
            x="Cell Type Percentage",
            bins=15,
            color="k",
            ax=ax[i + 1],
        )
        ax[i + 1].set(title=celltype, ylabel="# of Experiments")

    return f
