"""
This creates Figure 3.
"""
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, tensor_means


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)
    maxcluster = 5
    _, means, _ = probGMM(zflowDF, maxcluster, cellperexp)

    tMeans = tensor_means(zflowDF, means)

    rank = 5

    factors_PF,_ = tensor_decomp(tMeans, rank, "parafac")

    return f
