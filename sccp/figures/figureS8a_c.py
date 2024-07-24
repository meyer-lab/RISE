"""
Lupus: R2X for PCA/Pf2 and accuracy for different components
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
import anndata
import seaborn as sns
# from .commonFuncs.plotGeneral import plot_r2x
# from .commonFuncs.plotLupus import plot_accuracy_ranks_lupus


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 2), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    plot_cell_count_status(X, ax=ax[0])

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    # ranks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_r2x(X, ranks, ax[1])
    # plot_accuracy_ranks_lupus(X, ranks, ax[2], error_metric="accuracy")

    for i in [1, 2]:
        ax[i].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    return f


def plot_cell_count_status(X: anndata.AnnData, ax):
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
