"""
Figure 4a_c
"""

# from anndata import read_h5ad

from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    # X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    # plot_all_bulk_pred(X, ax[0], accuracy_metric="roc_auc")

    # ranks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_accuracy_ranks_lupus(X, ranks, ax[0], error_metric="roc_auc")
    ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    # X.uns["Pf2_A"] = correct_conditions(X)
    # plot_roc_fourthbatch(X, ax[1])
    # plot_labels_pacmap(X, "Cell Type2", ax[2])

    return f
