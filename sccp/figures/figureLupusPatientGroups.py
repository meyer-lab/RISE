"""
Lupus: Plotting Pf2 factors and weights
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
)
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)

    df = getSamplesObs(X.obs)

    plotConditionsFactors(X, ax[0], df["pool"], groupConditions=True)
    ax[0].set(yticks=[])
    plotConditionsFactors(X, ax[1], df["Processing_Cohort"], groupConditions=True)
    ax[1].set(yticks=[])
    plotConditionsFactors(X, ax[2], df["Status"], groupConditions=True)
    ax[2].set(yticks=[])

    return f
