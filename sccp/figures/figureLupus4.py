"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""

import numpy as np
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import plotR2X_pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    ranks = np.arange(5, 11, 5)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    plotR2X_pf2(X, ranks, ax[0])
    ax[0].set(xlim=(0, 55), ylim=(0, 0.15))

    return f
