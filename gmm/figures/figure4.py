"""
Investigating NK, covariance, and factors from tGMM for IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, markerslist, optimal_seed


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 300
    zflowTensor, _ = smallDF(cellperexp)
    rank = 3
    n_cluster = 3

    optimalseed, min_loglik = optimal_seed(
        1, zflowTensor, rank=rank, n_cluster=n_cluster
    )
    print(optimalseed)
    print(min_loglik)

    fac, x, _ = minimize_func(
        zflowTensor, rank=rank, n_cluster=n_cluster, seed=optimalseed
    )

    print("LogLik", x)

    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.norm_NK())
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    facXA = fac.get_factors_xarray(zflowTensor)

    for i, key in enumerate(facXA):
        data = facXA[key]
        sns.heatmap(
            data=data,
            xticklabels=data.coords[data.dims[1]].values,
            yticklabels=data.coords[data.dims[0]].values,
            vmin=0,
            ax=ax[i + 1],
        )

    # Covariance for different ranks
    for i in range(3):
        dff = pd.DataFrame(
            fac.covars[:, :, i] @ fac.covars[:, :, i].T,
            columns=markerslist,
            index=markerslist,
        )
        sns.heatmap(data=dff, ax=ax[i + 6])
        ax[i + 6].set(title="Covariance: Rank - " + str(i + 1))

    return f
