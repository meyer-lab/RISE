"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 24), (10, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/RawLupus.h5ad", backed="r")


    for i in range(50):
        print(i)
        plotCmpUMAP(X, i+1, ax[i], cbarMax=.2)
        # plotCmpPerCellType(X, i+1, ax[(2 * i)])
        # plotCmpUMAP(X, i+1, ax[(2 * i) + 1])

    return f
