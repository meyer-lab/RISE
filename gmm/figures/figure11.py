"""
# This creates Figure 11.
# """
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .common import subplotLabel, getSetup
from os.path import join, dirname
from ..CoHimport import CoH_xarray
from ..tensor import minimize_func, optimal_seed

path_here = dirname(dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20,30), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    
    cohXA = CoH_xarray(numCells=100,markers="Markers")
    print(cohXA)
    rank = 3
    n_cluster = 20

    optimalseed, _ = optimal_seed(5, cohXA, rank=rank, n_cluster=n_cluster)
    print(optimalseed)

    fac, x, _ = minimize_func(cohXA, rank=rank, n_cluster=n_cluster, seed=optimalseed)

    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.nk)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    fac_df = fac.get_factors_dataframes(cohXA)

    for i in range(0, 4):
        sns.heatmap(data=fac_df[i], vmin=0, ax=ax[i + 1])
    
    return f