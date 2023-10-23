"""
Lupus: Plot 2 Pf2 factors for conditions
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plot2DSeparationByComp
import numpy as np
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 8), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank=rank, dataName="Lupus")
    predict = "SLE_status"
    condStatus = X.obs[["Condition", predict]].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    df = pd.DataFrame(
        X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
        index=condStatus.index,
    )
    df = df.merge(condStatus, left_index=True, right_index=True)

    twoCmp = [[13, 26], [32, 26]]

    for i, pair in enumerate(twoCmp):
        plot2DSeparationByComp(df, pair, predict, ax[i])

    return f
