"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotGeneral import plotCV, plotR2X
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    rank = 30
    X = openPf2(rank, "Thomson")

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(factors, X, ax[0:3], reorder=(0, 2), trim=(2,))
    plotWeight(X.uns["Pf2_weights"], ax[3])

    # plotCV(X_pf, rank+1, trainPerc=0.75, ax=ax[4])
    # plotR2X(X_pf, rank+1, ax=ax[5])

    return f
