"""
Plots the differences in the identifying component weight with
different percentages of a cell type removed from the data.
The determining component is the component that has the highest
correlation with the number of cells in the cell type selected.
"""

from .common import getSetup
from ..imports import import_thomson
import numpy as np
from ..factorization import pf2
from anndata import AnnData
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


def makeFigure():
    rank = 20
    data = import_thomson()
    ax, f = getSetup((2, 2), (1, 1))

    plot_weights_across_percents(data, "B Cells", "CTRL4", 0, 1, 0.25, rank, ax[0])

    ### Can add other cell types here

    return f


def plot_weights_across_percents(
    data: AnnData,
    cell_type: str,
    condition: str,
    percent_min: float,
    percent_max: float,
    percent_step: float,
    rank: int,
    ax: Axes,
    ct2: bool = False,
    override: int = -1,
):
    """
    Plots the raw weight of the identifying component across the percentages dropped
    """
    all_percents: list[int] = []
    all_weights: list[float] = []
    ctarg = "Cell Type2" if ct2 else "Cell Type"
    yt = pd.Series(np.unique(data.obs["Condition"]))
    numberOfCellType = [
        len(data[(data.obs["Condition"] == txt) & (data.obs[ctarg] == cell_type)])
        for txt in yt
    ]  # Number of cells in the chosen condition

    for _ in range(5):
        vals = {}
        for percent in np.arange(percent_min, percent_max + percent_step, percent_step):
            idx = (data.obs["Cell Type"] != cell_type) | (
                data.obs["Condition"] != condition
            )

            # Calculate the number of cells to drop based on the percentage
            false_idx = idx[~idx].index
            size = int(len(false_idx) * (1 - percent))

            idx[np.random.choice(false_idx, size=size, replace=False)] = True
            sampled_data = data[idx]

            sampledX = pf2(sampled_data, rank, doEmbedding=False)

            if override == -1:  # Use r^2 values to find the most important component
                X = np.array(sampledX.uns["Pf2_A"])
                all_r2 = [
                    linregress(X[:, i], numberOfCellType)[2] ** 2
                    for i in range(X.shape[1])
                ]
                most_exp_cmp = int(np.argmax(all_r2))
            else:  # Use the override component numbers
                most_exp_cmp = override

            Y = np.array(sampledX.uns["Pf2_A"])[:, most_exp_cmp]
            idx = np.where(np.unique(sampledX.obs["Condition"]) == condition)[0][
                0
            ]  # Aims to find the index of the condition to set its corresponding weight
            vals[percent] = Y[idx]

        percents, weights = zip(*sorted((int(p * 100), w) for p, w in vals.items()))
        all_percents += percents
        all_weights += weights

    df = pd.DataFrame(data={"Percent": all_percents, "Weight": all_weights})
    df = df.sort_values(by="Percent")
    sns.lineplot(data=df, x="Percent", y="Weight", ax=ax)
    ax.set_xlabel("% B Cells Dropped")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_ylabel(f"{cell_type} Component Weight ")
