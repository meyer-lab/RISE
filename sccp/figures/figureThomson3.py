"""
Thomson: PaCMAP of weighted projections 
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import (
    plot_wp_pacmap,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    for i in range(1, 21):
        plot_wp_pacmap(X, i, ax[i-1], cbarMax=.3)

    return f
