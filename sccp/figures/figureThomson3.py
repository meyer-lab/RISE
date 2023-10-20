"""
Thomson: Plotting weighted projections per component
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, "Thomson")

    for i in range(0, 8):
        plotCmpUMAP(X, i, ax[i])

    return f
