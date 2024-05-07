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
    ax, f = getSetup((5, 2), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    # ranks = np.arange(5, 10, 5)
    ranks=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_r2x(X, ranks, ax[0])

    
    
    plot_predaccuracy_ranks_lupus(X, ranks, ax[1], error_metric="roc_auc")
    ax[1].set(
        xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    X = [0, 50]
    Y = [.84, .84]
 
    # Plot a graph
    ax[1].plot(X, Y, linestyle='--')


    return f
