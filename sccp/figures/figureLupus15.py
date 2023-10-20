"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, openPf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = openPf2(rank=40, dataName="Lupus")

    df = X.obs[["Cell Type", "SLE_status", "Condition"]].reset_index(drop=True)

    dfCond = df.groupby(["Condition"]).size().reset_index(name="Cell Number")
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")

    dfCellType = df.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count")
    for i, cond in enumerate(pd.unique(df["Condition"])):
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
