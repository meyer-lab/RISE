"""
This creates Figure 4.
"""
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM, meanmarkerDF
from ..tensor import tensor_decomp, tensor_means, tensor_covar, meanCP_to_DF, covarTens_to_DF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 500
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 7
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)

    meansDF, markerslist = meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster)

    # tensor_means(DF,means of markers): [tensor form of means] converts DF into tensor
    tMeans = tensor_means(zflowDF, means)

    # tensor_covar((DF,covariance of markers): [tensor form of covarinaces] converts DF into tensor
    _ = tensor_covar(zflowDF, covar)

    # tensor_decomp(tensor means, rank, type of decomposition): [DF,tensorfactors/weights] creates DF of factors for different conditions and output of decomposition
    rank = 5
    factors_NNP, factorinfo_NNP = tensor_decomp(tMeans, rank, "NNparafac")

    # meanCP_to_DF(factors/weights,short DF):[DF] converts tensor decomposition to DF
    markDF = meanCP_to_DF(factorinfo_NNP, zflowDF)
    print(markDF)

    # covarTens_to_DF(DF,covariances,list of all markers):[DF] converts output of GMM to DF
    covarDF = covarTens_to_DF(meansDF, covar, markerslist)
    print(covarDF)

    return f
