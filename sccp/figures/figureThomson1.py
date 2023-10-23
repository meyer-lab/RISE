"""
Thomson: Plotting Pf2 factors and weights
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

    rank = 30
    X = openPf2(rank, "Thomson")
    print(X.shape)

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(factors, X, ax[0:3], reorder=(0, 2), trim=(2,))
    plotWeight(X.uns["Pf2_weights"], ax[3])

    return f
