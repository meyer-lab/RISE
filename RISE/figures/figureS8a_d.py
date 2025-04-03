"""
Figure 8a_d
"""

import anndata
import seaborn as sns
from matplotlib.axes import Axes

from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    plot_cell_count_status(X, ax=ax[0])

    # ranks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_r2x(X, ranks, ax[1])

    # plot_all_bulk_pred(X, ax[2], accuracy_metric="accuracy")
    # plot_accuracy_ranks_lupus(X, ranks, ax[3], error_metric="roc_auc", bootstrap=True)

    for i in [1, 2, 3]:
        ax[i].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    return f


def plot_cell_count_status(X: anndata.AnnData, ax: Axes):
    """Plots overall cell count for SLE and healthy patients"""
    df = X.obs[["SLE_status", "Condition"]].reset_index(drop=True)
    dfCond = (
        df.groupby(["Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )

    sns.boxplot(
        data=dfCond,
        x="SLE_status",
        y="Cell Count",
        hue="SLE_status",
        showfliers=False,
        ax=ax,
    )
