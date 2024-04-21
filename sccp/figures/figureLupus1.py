"""
Lupus: Plotting Pf2 factors and weights
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
    plotWeight,
)
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    lupusStatus = getSamplesObs(X.obs)["SLE_status"]

    X.uns["Pf2_A"] = correct_conditions(X)

    plotConditionsFactors(X, ax[0], lupusStatus)
    ax[0].set(yticks=[])
    plotCellState(X, ax[1])
    plotGeneFactors(X, ax[2])
    plotWeight(X, ax[3])

    return f
