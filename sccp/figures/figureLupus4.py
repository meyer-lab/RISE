"""
Lupus: R2X for PCA/Pf2 and prediction accuracy for different components
"""

import numpy as np
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import plot_r2x
from .commonFuncs.plotLupus import plot_predaccuracy_ranks_lupus


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 4), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    ranks = np.arange(5, 10, 5)
    plot_r2x(X, ranks, ax[0])

    plot_predaccuracy_ranks_lupus(X, ranks, ax[1])

    return f
