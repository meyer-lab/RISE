"""
Thomson: Plotting weighted projections per component
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    for i in range(1, 21):
        plotCmpUMAP(X, i, ax[i-1], cbarMax=.3)

    return f
