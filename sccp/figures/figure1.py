"""
Creating synthetic data and running ULTRA to calculate factors and recapitulated moving covariance
"""
import os
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports.scRNA import ThompsonXA_RawGenes
from tensorly.decomposition import parafac2

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Gene, Cell, Drug]
    drugXA = ThompsonXA_RawGenes()

    # Performing parafac2 on single-cell Xarray
    rank = 5
    weights, factors, _ = parafac2(
        drugXA.to_numpy(),
        rank=rank,
        tol=1e-10,
        nn_modes=(0, 2),
        normalize_factors=True,
        verbose=True
    )

    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for i in range(0, 3):
        sns.heatmap(
            data=factors[i],
            xticklabels=xticks,
            yticklabels=drugXA.coords[drugXA.dims[i]].values,
            ax=ax[i],
            cmap=cmap,
            vmax=1,
            vmin=-1,
        )

        ax[i].set_title("Mean Factors")
        ax[i].tick_params(axis="y", rotation=0)

    return f
