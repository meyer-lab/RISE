"""
Thomson dataset: Cell counts and cell type percentages per condition.
"""
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

    rank = 30
    X = openPf2(rank, "Thomson")

    df = pd.DataFrame(
        {"Cell Type": gateThomsonCells(X), "Condition": X.obs["Condition"]}
    )

    # Per condition counts
    dfCond = df.groupby(["Condition"]).size().reset_index(name="Cell Number")
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")

    # Per condition cell type percentages
    dfCellType = df.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count")
    dfCellType["Cell Type Percentage"] = (
        100
        * dfCellType["Count"]
        / dfCellType.groupby("Condition")["Count"].transform("sum")
    )

    for i, (name, group) in enumerate(dfCellType.groupby("Cell Type")):
        sns.histplot(
            data=group,
            x="Cell Type Percentage",
            bins=15,
            color="k",
            ax=ax[i + 1],
        )
        ax[i + 1].set(title=name, ylabel="# of Experiments", xlim=(0.0, group["Cell Type Percentage"].max()))

    return f
