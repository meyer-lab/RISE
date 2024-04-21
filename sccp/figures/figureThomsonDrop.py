"""
Describes the model's reaction to the removal of specific pieces of the
dataset. It can clearly be observed for all cell types that had an identifying component that the
corresponding weight for the condition removed significantly dropped.
"""

from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.stats import linregress


def makeFigure():
    rank = 20
    data = import_thomson()

    ax, f = getSetup((6, 3), (1, 2))

    plotDifferentialExpression(data, "CTRL4", "B Cells", rank, ax[0:2])

    ### Uncomment to plot other cell types

    # plotDifferentialExpression(
    #     data,
    #     "CTRL4",
    #     "Naive B Cells",
    #     rank,
    #     *ax[2:4],
    #     ct2=True,
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "NK Cells", rank, *ax[4:6]
    # )
    # plotDifferentialExpression(data, "CTRL4", "DCs", rank, *ax[6:8])
    # plotDifferentialExpression(
    #     data,
    #     "CTRL4",
    #     "pDCs",
    #     rank,
    #     *ax[8:10],
    #     ct2=True,
    # )
    # plotDifferentialExpression(data, "CTRL4", "T Cells", rank, *ax[10:12])
    # plotDifferentialExpression(
    #     data, "CTRL4", "Cytotoxic T Cells", rank, *ax[12:14], ct2=True
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "Memory T Cells", rank, *ax[14:16], ct2=True
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "cDCs", rank, *ax[16:18], ct2=True, override=(14, 15)
    # )
    return f


def plotDifferentialExpression(
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
    ]  # Number of cells in the chosen condition

    if override == (-1, -1):  # Use r^2 values to find the most important component
        X, X2 = np.array(origX.uns["Pf2_A"]), np.array(sampledX.uns["Pf2_A"])
        all_r2, all_r2_2 = (
            [linregress(X[:, i], numberOfCellType)[2] ** 2 for i in range(X.shape[1])],
            [linregress(X2[:, i], numberOfCellType)[2] ** 2 for i in range(X.shape[1])],
        )
        most_exp_cmp, most_exp_cmp2 = int(np.argmax(all_r2)), int(np.argmax(all_r2_2))
    else:  # Use the override component numbers
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
    axes[0].set_xlabel(f"Component {most_exp_cmp + 1} Weight (Full Data)")
    axes[0].set_ylabel(f"Component {most_exp_cmp + 1} Weight (Sampled Data)")
    axes[0].set_title(f"Full Data vs {cell_type} removed from {condition}")
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(left=0)
    axes[0].legend(loc="upper left")

    axes[1].set_xlabel(f"Component {most_exp_cmp + 1} Weight (Full Data)")
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
