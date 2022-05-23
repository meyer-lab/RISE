"""
This creates Figure 5.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, tensorGMM_CV


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 100
    ranknumb = np.arange(3, 11)
    n_cluster = np.arange(4, 8)

    zflowTensor, _ = smallDF(cellperexp)
    # maxloglikDF = pd.DataFrame(columns=["Rank", "Cluster", "MaxLoglik"])
    maxloglikDF = pd.DataFrame()
    for i in range(len(ranknumb)):
        row = pd.DataFrame()
        row["Rank"] = ["Rank:" + str(ranknumb[i])]
        for j in range(len(n_cluster)):
            _, _, _, loglik, _ = minimize_func(zflowTensor, ranknumb[i], n_cluster[j])
            row["Cluster:" + str(n_cluster[j])] = loglik

        maxloglikDF = pd.concat([maxloglikDF, row])

    maxloglikDF = maxloglikDF.set_index("Rank")
    sns.heatmap(data=maxloglikDF, ax=ax[0])

    maxloglikDFcv = pd.DataFrame()
    for i in range(len(ranknumb)):
        row = pd.DataFrame()
        row["Rank"] = ["Rank:" + str(ranknumb[i])]
        for j in range(len(n_cluster)):
            loglik = tensorGMM_CV(zflowTensor, numFolds=3, numClusters=n_cluster[j], numRank=ranknumb[i])
            row["Cluster:" + str(n_cluster[j])] = loglik

        maxloglikDFcv = pd.concat([maxloglikDFcv, row])

    maxloglikDFcv = maxloglikDFcv.set_index("Rank")
    sns.heatmap(data=maxloglikDFcv, ax=ax[1])
    ax[1].set(title="Cross Validation")

    return f
