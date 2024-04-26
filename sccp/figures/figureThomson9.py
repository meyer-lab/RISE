"""
Thomson: Weighted projections per cell type
"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_per_celltype


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 14), (5, 4))

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_wp_per_celltype(X, i + 1, ax[i], outliers=False, cellType="Cell Type2")

    return f
