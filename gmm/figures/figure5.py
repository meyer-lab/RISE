"""
This creates Figure 5.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from jax.config import config
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func


config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    ranknumb = np.arange(1, 6)
    n_cluster = np.arange(2, 8)
    zflowTensor, _ = smallDF(cellperexp)

    # maxloglikDF = pd.DataFrame(columns=["Rank", "Cluster", "MaxLoglik"])
    maxloglikDF = pd.DataFrame()

    for i in range(len(ranknumb)):
        row = pd.DataFrame()
        row["Rank"] = ["Rank:" + str(ranknumb[i])]
        for j in range(len(n_cluster)):
            _, _, _, loglik = minimize_func(zflowTensor, ranknumb[i], n_cluster[j], maxiter=1000)
            row["Cluster:" + str(n_cluster[j])] = loglik

        maxloglikDF = pd.concat([maxloglikDF, row])

    maxloglikDF = maxloglikDF.set_index("Rank")
    sns.heatmap(data=maxloglikDF, ax=ax[0])

    return f
