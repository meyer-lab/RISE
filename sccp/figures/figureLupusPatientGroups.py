"""
Lupus: Plotting Pf2 factors and weights
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
)
from .commonFuncs.plotLupus import getSamplesObs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    df = getSamplesObs(X.obs)

    plotConditionsFactors(X, ax[0], df["pool"])
    plotConditionsFactors(X, ax[1], df["Processing_Cohort"])
    plotConditionsFactors(X, ax[2], df["Status"])

    return f
