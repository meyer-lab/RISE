"""
Creating synthetic data and running ULTRA to calculate factors and recapitulated moving covariance
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
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


def ThompsonXA_RawGenes():
    """Turns filtered and normalized cells into an Xarray."""
    df = pd.read_csv("/opt/andrew/FilteredLogDrugs_Offset_1.1.csv", sep=",")
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    df = df.sort_values(by=["Drug"])

    # Assign cells a count per-experiment so we can reindex
    cellCount = df.groupby(by=["Drug"]).size().values
    df["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in cellCount])

    xarr = df.set_index(["Cell", "Drug"]).to_xarray()
    xarr = xarr.to_array(dim="Gene")

    ### I *believe* that padding with zeros does not affect PARAFAC2 results.
    ### We should check this though.
    xarr.values = np.nan_to_num(xarr.values)

    return xarr
