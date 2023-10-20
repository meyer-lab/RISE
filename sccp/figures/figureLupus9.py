"""
S3d: Plot samples along two components to see patient separation
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: see if SLE/healthy samples can be stratified along strongly predictive Pf2 components
# (they can, at least when you do 13 and 26)

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plot2DSeparationByComp
import numpy as np
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 8), (1, 2))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank=40
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
