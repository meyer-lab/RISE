"""
CITEseq: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
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

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")

    for i, axi in enumerate(ax):
        plotCmpUMAP(X, i + 1, axi)

    return f
