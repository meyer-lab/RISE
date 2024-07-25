"""
Figure S9: PCA and Pf2 PaCMAP labeled by genes and drugs PaCMAP for all components
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 10), (6, 5))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    for i in range(1, 31):
        plot_wp_pacmap(X, i, ax[i - 1], cbarMax=0.3)

    return f
