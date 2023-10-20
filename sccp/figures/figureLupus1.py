"""
S1: Initial Attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: test Pf2 on lupus data, get visualizations for factor matrices

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2, savePf2
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
from ..imports.scRNA import import_lupus
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 16), (2, 2))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    # X = import_lupus()
    # X = pf2(X, "Condition", rank=rank)
    # savePf2(X, rank, "Lupus")
    X = openPf2(rank, "Lupus")
    lupusStatus = X.obs[["Condition", "SLE_status"]].drop_duplicates("Condition")
    lupusStatus = lupusStatus.set_index("Condition")["SLE_status"]

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(
        factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=lupusStatus
    )
    plotWeight(X.uns["Pf2_weights"], ax[3])

    # X_pf = tensorFy(X, "Condition")
    # plotCV(X_pf, rank+1, trainPerc=0.75, ax=ax[4])
    # plotR2X(X_pf, rank+1, ax=ax[5])

    return f
