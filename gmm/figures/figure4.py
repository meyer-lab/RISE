"""
This creates Figure 4.
"""
import numpy as np
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
    zflowTensor, _ = smallDF(cellperexp)

    maximizedNK, maximizedFactors, _ = minimize_func(zflowTensor, rank=3, n_cluster=6)

    ax[0].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 1])

    return f
