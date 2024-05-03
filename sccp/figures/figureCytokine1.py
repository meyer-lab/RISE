"""
Cytokines: Plotting Cytokine factors and weights
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
    plot_factor_weight,
)
from .commonFuncs.plotLupus import samples_only_lupus
from ..factorization import correct_conditions
# from ..imports import import_cytokine
# from ..factorization import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)
    """
    X = import_cytokine()
    ranks = [10, 20, 30]
    for rank in ranks:
        cytok_pf2 = pf2(X, rank)
        cytok_pf2.write_h5ad("/home/brianoj/scCP/cytok_pf2/cytok_pf2_" + str(rank))
    """
    X = read_h5ad("sccp/notebooks/cytok/cytok_pf2_30.h5ad")

    X.uns["Pf2_A"] = correct_conditions(X)
    stimulations = samples_only_lupus(X)["Condition"]

    plot_condition_factors(X, ax[0], stimulations)
    ax[0].set(yticks=[])
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    plot_factor_weight(X, ax[3])

    return f
