"""
Figure S7
"""

import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.stats import linregress
import seaborn as sns
from .common import getSetup, subplotLabel
from ..imports import import_thomson
from ..factorization import pf2


def makeFigure():
    ax, f = getSetup((5, 5), (2, 2))
    subplotLabel(ax)

    rank = 20
    data = import_thomson()

    plot_weights_across_percents(data, "B Cells", "CTRL4", 0, 1, 0.25, rank, ax[3])
    plot_diff_exp(data, "CTRL4", "B Cells", rank, ax[0:2])

    return f


def plot_diff_exp(
    data: AnnData,
    condition: str,
    cell_type: str,
    rank: int,
    axes: list[Axes],
    ct2: bool = False,
    override: tuple[int, int] = (-1, -1),
):
    """
    Plots the differences in model weights of a gene set in the original data and the data. The determining component is the component that has the highest correlation with the number of cells in the cell type selected.
    Args:
        data: The AnnData object to be used
        condition: The condition to drop the cell type from
        cell_type: The cell type to be drop
        geneset: The marker genes
        rank: The rank to be used
        *args: The axes to be used
        ct2: Whether to use the second cell type
        override: The component to be used if not using the highest absolute value weight
    Returns:
        None
    """

    ctarg = "Cell Type2" if ct2 else "Cell Type"
    sampled_data = data[
        (data.obs[ctarg] != cell_type) | (data.obs["Condition"] != condition)
    ]

    origX = pf2(data, rank, doEmbedding=False)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)
    yt = pd.Series(np.unique(origX.obs["Condition"]))
    numberOfCellType = [
        len(data[(data.obs["Condition"] == txt) & (data.obs[ctarg] == cell_type)])
        for txt in yt
    ]

    if override == (-1, -1):  # Use R^2 values to find  most important component
        X, X2 = np.array(origX.uns["Pf2_A"]), np.array(sampledX.uns["Pf2_A"])
        all_r2, all_r2_2 = (
            [linregress(X[:, i], numberOfCellType)[2] ** 2 for i in range(X.shape[1])],
            [
                linregress(X2[:, i], numberOfCellType)[2] ** 2
                for i in range(X2.shape[1])
            ],
        )
        most_exp_cmp, most_exp_cmp2 = int(np.argmax(all_r2)), int(np.argmax(all_r2_2))
    else:  # Uses specific component number
        most_exp_cmp, most_exp_cmp2 = override[0], override[1]
        all_r2 = [0] * rank
        all_r2[most_exp_cmp] = (
            linregress(np.array(origX.uns["Pf2_A"])[:, most_exp_cmp], numberOfCellType)[
                2
            ]
            ** 2
        )

    X, Y = (
        np.array(origX.uns["Pf2_A"])[:, most_exp_cmp],
        np.array(sampledX.uns["Pf2_A"])[:, most_exp_cmp2],
    )

    colors = ["r" if txt == condition else "b" for txt in yt]

    axes[0].scatter(X, Y, c=colors)
    a, b = np.polyfit(X, Y, 1)
    axes[0].axline((0, b), slope=a, linestyle="--")
    axes[0].scatter([], [], c="b", label="Other Conditions")
    axes[0].scatter([], [], c="r", label=condition)
    axes[0].set_xlabel(f"{cell_type} Component Weight (Full Data)")
    axes[0].set_ylabel(f"{cell_type} Component Weight (Sampled Data)")
    axes[0].set_title(f"Full Data vs {cell_type} removed from {condition}")
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(left=0)
    axes[0].legend(loc="upper left")

    axes[1].set_xlabel(f"{cell_type} Component Weight Weight (Full Data)")
    axes[1].set_ylabel(f"Number of {cell_type} per condition (Full Data)")
    axes[1].scatter(X, numberOfCellType)
    a, b = np.polyfit(X, numberOfCellType, 1)
    axes[1].axline(
        (0, b),
        slope=a,
        label=f"R^2 value: {int(all_r2[most_exp_cmp]*100)/100}",
        linestyle="--",
    )
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlim(left=0)
    axes[1].legend(loc="upper left")
    print(f"most_exp_cmp: {most_exp_cmp}, most_exp_cmp2: {most_exp_cmp2}")


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
