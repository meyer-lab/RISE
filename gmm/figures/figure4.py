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
        3, zflowTensor, rank=rank, n_cluster=n_cluster
    )
    print(optimalseed)
    print(min_loglik)

    fac, x, _ = minimize_func(
        zflowTensor, rank=rank, n_cluster=n_cluster, seed=optimalseed
    )
    ptMarkerPatterns = fac.covars

    print("LogLik", x)

    for i in range(3):
        dff = pd.DataFrame(
            ptMarkerPatterns[:, :, i] @ ptMarkerPatterns[:, :, i].T,
            columns=markerslist,
            index=markerslist,
        )
        sns.heatmap(data=dff, ax=ax[i])
        ax[i].set(title="Covariance: Rank - " + str(i))

    ax[3].bar(np.arange(1, fac.nk.size + 1), fac.nk)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    fac_df = fac.get_factors_dataframes(zflowTensor)

    for i in range(0, len(fac_df)):
        sns.heatmap(data=fac_df[i], vmin=0, ax=ax[i + 4])

    return f
