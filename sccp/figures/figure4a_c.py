"""
Figure 4a_c
"""

from anndata import read_h5ad

from ..factorization import correct_conditions
from .common import getSetup, subplotLabel
from .commonFuncs.plotLupus import plot_roc_fourthbatch
from .commonFuncs.plotPaCMAP import plot_labels_pacmap

# from .commonFuncs.plotLupus import plot_accuracy_ranks_lupus
# from .commonFuncs.plotGeneral import plot_r2x


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((9, 4), (1, 3))
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    # ranks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_accuracy_ranks_lupus(X, ranks, ax[0], error_metric="roc_auc")
    ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    x = [0, 50]
    y = [0.84, 0.84]
    ax[0].plot(x, y, linestyle="--")

    X.uns["Pf2_A"] = correct_conditions(X)
    plot_roc_fourthbatch(X, ax[1])

    plot_labels_pacmap(X, "Cell Type2", ax[2])

    return f
