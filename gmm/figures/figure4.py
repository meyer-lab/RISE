"""
This creates Figure 4.
"""
from scipy.stats import gmean
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, tensor_covar, meanCP_to_DF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 10
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 4
    nk, tMeans, covar = probGMM(zflowDF, maxcluster, cellperexp)

    conditions = zflowDF.iloc[::cellperexp]
    conditions = conditions[["Time", "Dose", "Ligand"]]
    conditions = conditions.set_index(["Time", "Dose", "Ligand"])

    # Tensorify data
    tCovar = tensor_covar(conditions, covar)

    # tensor_decomp(tensor means, rank, type of decomposition):
    # [DF,tensorfactors/weights] creates DF of factors for different
    # conditions and output of decomposition
    rank = 5
    factors_NNP, factorinfo_NNP = tensor_decomp(tMeans, rank, "NNparafac")

    # meanCP_to_DF(factors/weights,short DF):[DF] converts tensor decomposition to DF
    markDF = meanCP_to_DF(factorinfo_NNP, tMeans)

    nkCommon = gmean(nk, axis=(1, 2, 3)) # nk is shared across conditions

    return f
