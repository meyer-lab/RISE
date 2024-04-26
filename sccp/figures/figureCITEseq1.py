"""
CITEseq: Pf2 factors, weights, PaCMAP labeled by all conditions/leiden clusters,
and ratio of condition components based on days
"""

from anndata import read_h5ad
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
    plot_factor_weight,
)
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    plot_condition_factors(X, ax[0])
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    plot_factor_weight(X, ax[3])

    plot_labels_pacmap(X, "Condition", ax[4])
    # plot_labels_pacmap(X, "leiden", ax[5])
    plot_condition_factor_ratio(X, ax[6], day7=False)
    plot_condition_factor_ratio(X, ax[7], day7=True)

    return f


def plot_condition_factor_ratio(X: anndata.AnnData, ax: Axes, day7=True):
    """Plots ratio of condition factors for day 1 or 7"""
    p = np.unique(X.obs["Condition"])
    p = [p[2], p[1], p[0], p[3], p[4]]
    d = X.uns["Pf2_A"]
    X = np.array([d[2], d[1], d[0], d[3], d[4]])
    xticks = np.arange(1, np.shape(X)[1] + 1)

    if day7 is False:
        ratio = X[1, :] / X[-2, :]
        day = 7
        yticks = [0.45, 1, 1.55]

    else:
        ratio = X[0, :] / X[-1, :]
        day = 1
        yticks = [2, 1.5, 1, 0.5, 0]

    ax.plot(xticks, ratio)
    ax.set(
        xticks=np.arange(1, np.shape(X)[1] + 1, 2),
        yticks=yticks,
        xlabel="Component",
        ylabel=(f"IC/SC Ratio Day {day}"),
    )
