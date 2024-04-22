"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (6, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    for i in range(1, 31):
        plotCmpUMAP(X, i, ax[i-1], cbarMax=.3)
        
        
    return f
