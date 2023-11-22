"""
Lupus: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    lupusStatus = X.obs[["Condition", "SLE_status"]].drop_duplicates("Condition")
    lupusStatus = lupusStatus.set_index("Condition")["SLE_status"]

    plotFactors(X, ax[0:3], reorder=(0, 2), cond_group_labels=lupusStatus)
    plotWeight(X.uns["Pf2_weights"], ax[3])

    return f
