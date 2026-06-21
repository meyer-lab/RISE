"""
Figure 8g_h
"""

import anndata

from analysis.figures.commonFuncs.plotLupus import samples_only_lupus
from RISE.factorization import correct_conditions
from RISE.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
)
from RISE.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap

from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    lupusStatus = samples_only_lupus(X)["SLE_status"]

    X.uns["Pf2_A"] = correct_conditions(X)

    plot_condition_factors(X, ax[0], cond_group_labels=lupusStatus)
    ax[0].set(yticks=[])
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    ax[2].yaxis.set_ticklabels([])

    plot_labels_pacmap(X, "Cell Type", ax[3])

    return f
