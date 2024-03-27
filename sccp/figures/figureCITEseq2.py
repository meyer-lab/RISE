"""
CITEseq: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotCmpUMAP, plotCmpPerCellType


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 20), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    comps = [22, 33, 47, 48, 23, 31, 43]

    for i, cmp in enumerate(comps):
        plotCmpPerCellType(X, cmp, ax[2 * i], cellType="leiden")
        plotCmpUMAP(X, cmp, ax[2 * i + 1], cbarMax=0.25)


    return f
