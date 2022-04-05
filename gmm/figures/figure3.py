"""
This creates Figure 3.
"""
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, tensor_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 5
    _, tMeans, _ = probGMM(zflowDF, maxcluster)

    # tensor_R2X(tensor means, maximum rank): [list of rankings,varexpl_NNP] outputs  and variance explained
    maxrank = 10
    rankings, varexpl_NNP = tensor_R2X(tMeans, maxrank, "NNparafac")

    ax[0].plot(rankings, varexpl_NNP, "r")
    xlabel = "Number of Components"
    ylabel = "R2X"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # tensor_decomp(tensor means, rank, type of decomposition): [DF,tensor
    # factors/weights] creates DF of factors for different conditions and
    # output of decomposition
    rank = 5
    factors_NNP, _ = tensor_decomp(tMeans, rank, "NNparafac")

    for i in range(0, 5):
        heatmap = sns.heatmap(data=factors_NNP[i], ax=ax[i + 1], vmin=0, vmax=1, cmap="Blues")

    return f
