"""
Thomson: Plotting weighted projections per component
"""
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    for i in range(0, 8):
        plotCmpUMAP(X, i, ax[i])

    return f
