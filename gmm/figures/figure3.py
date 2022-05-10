"""
This creates Figure 3.
"""
import seaborn as sns

from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.GMM import probGMM
from gmm.tensor import tensor_decomp, tensor_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 50
    zflowDF, _ = smallDF(cellperexp)

    # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 3
    _, tMeans, _ = probGMM(zflowDF, maxcluster)

    # tensor_R2X(tensor means, maximum rank): [list of rankings,varexpl_NNP] ranking and variance explained
    maxrank = 6
    rankings, varexpl_NNP = tensor_R2X(tMeans, maxrank)

    ax[0].plot(rankings, varexpl_NNP, "r")
    xlabel = "Number of Components"
    ylabel = "R2X"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # tensor_decomp(tensor means, rank, type of decomposition): [DF,tensor
    # factors/weights] creates DF of factors for different conditions and
    # output of decomposition
    rank = 6
    factors_NNP, facInfo = tensor_decomp(tMeans, rank)

    for i in range(0, len(facInfo.shape)):
        heatmap = sns.heatmap(data=factors_NNP[i], ax=ax[i + 1], vmin=0, vmax=1, cmap="Blues")

    return f
