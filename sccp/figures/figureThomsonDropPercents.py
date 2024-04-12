"""
Plots the differences in the identifying component weight with different percentages of a cell type removed from the data. The determining component is the component that has the highest correlation with the number of cells in the cell type selected.
"""

from .common import getSetup
from ..imports import import_thomson
import numpy as np
from ..factorization import pf2
import pandas as pd
import seaborn as sns


def makeFigure():
    rank = 20
    data = import_thomson()
    ax, f = getSetup((2, 2), (1, 1))
    bCellGeneSet = [
        "PXK",
        "MS4A1",
        "CD19",
        "CD74",
        "CD79A",
        "CD79B",
        "BANK1",
        "PTPRC",
        "CR2",
        "VPREB3",
    ]

    plot_weights_across_percents(
        data, "B Cells", "CTRL4", 0, 1, 0.1, rank, bCellGeneSet, ax[0]
    )

    ### Can add other cell types here

    return f


def plot_weights_across_percents(
    data,
    cell_type,
    condition,
    percent_min,
    percent_max,
    percent_step,
    rank,
    geneset,
    ax,
):
    """
    Plots the raw weight of the identifying component across the percentages dropped
    """
    all_percents, all_weights = [], []
    for i in range(5):
        vals = {}
        for percent in np.arange(percent_min, percent_max, percent_step):
            idx = (data.obs["Cell Type"] != cell_type) | (
                data.obs["Condition"] != condition
            )

            # Calculate the number of cells to drop based on the percentage
            false_idx = idx[~idx].index
            size = int(len(false_idx) * (1 - percent)) 

            idx[np.random.choice(false_idx, size=size, replace=False)] = True
            sampled_data = data[idx]

            sampledX = pf2(sampled_data, rank, doEmbedding=False)
            gene_values = np.array(sampledX.varm["Pf2_C"])[
                [i for i, gene in enumerate(data.var.index.values) if gene in geneset]
            ] # Sum of the gene expression for the marker genes
            most_exp_cmp = np.argmax(np.sum(np.abs(gene_values), axis=0)) # The component with the highest sum of gene expression
            Y = np.array(sampledX.uns["Pf2_A"])[:, most_exp_cmp]
            idx = np.where(np.unique(sampledX.obs["Condition"]) == condition)[0][0]  # Aims to find the index of the condition to set its corresponding weight
            vals[percent] = Y[idx]

        percents, weights = zip(*sorted((int(p * 100), w) for p, w in vals.items()))
        all_percents += percents
        all_weights += weights

    df = pd.DataFrame(data={"Percent": all_percents, "Weight": all_weights})
    df = df.sort_values(by="Percent")
    sns.lineplot(data=df, x="Percent", y="Weight", ax=ax)
    ax.set_xlabel("% B Cells Dropped")
    ax.set_ylabel(f"Component {most_exp_cmp} Weight")
