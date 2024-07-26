"""
Figure S6
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 10), (5, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")

    for i in range(1, 21):
        plot_wp_pacmap(X, i, ax[i - 1], cbarMax=0.3)

    return f
