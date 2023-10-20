"""
CITEseq: Plotting weighted projections per component
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = import_citeseq()
    X = pf2(X, "Condition", rank=40)

    for i in range(25):
        plotCmpUMAP(X, i + 1, ax[i])

    return f
