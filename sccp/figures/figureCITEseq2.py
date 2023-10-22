"""
CITEseq: Plotting weighted projections per component
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = openPf2(80, "CITEseq")

    for i, axi in enumerate(ax):
        plotCmpUMAP(X, i + 1, axi)

    return f
