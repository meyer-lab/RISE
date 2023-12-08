"""
Lupus: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 28), (8, 5))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    for i in range(3):
        plotCmpPerCellType(X, i + 1, ax[i], outliers=False)

    return f
