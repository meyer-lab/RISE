"""
This creates Figure 11.
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..CoHimport import CoH_xarray
from ..tensor import optimal_seed


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    cohXA = CoH_xarray(numCells=100, markers="Markers")

    rank = 3
    n_cluster = 20

    _, _, fit = optimal_seed(10, cohXA, rank=rank, n_cluster=n_cluster)
    fac = fit[0]

    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.nk)
    ax[0].set(xlabel="Cluster", ylabel="NK Value")

    # CP factors
    facXA = fac.get_factors_xarray(cohXA)

    for i, key in enumerate(facXA):
        if i < 4:
            data = facXA[key]
            sns.heatmap(
                data=data,
                xticklabels=data.coords[data.dims[1]].values,
                yticklabels=data.coords[data.dims[0]].values,
                ax=ax[i + 1])

    return f
