"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    comps = [4, 7, 8, 10, 22, 24]
    for i, cmp in enumerate(comps):
        plotCmpPerCellType(X, cmp, ax[(2 * i)])
        plotCmpUMAP(X, cmp, ax[(2 * i) + 1], cbarMax=0.3)
    return f
