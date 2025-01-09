"""
Figure 8d_e
"""

import anndata

from .common import getSetup, subplotLabel
from .commonFuncs.plotLupus import plot_roc_fourthbatch


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    # ranks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # plot_accuracy_ranks_lupus(X, ranks, ax[3], error_metric="roc_auc",
    #                           bootstrap=False, cv_fourth_batch=False)

    plot_roc_fourthbatch(X, ax[1], cv_fourth_batch=False)

    ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    return f
