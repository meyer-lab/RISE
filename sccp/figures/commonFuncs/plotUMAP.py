import seaborn as sns
from matplotlib import pyplot as plt
import umap.plot
import umap
import numpy as np
import pandas as pd


def plotCondUMAP(conds, decomp, totalconds, points, axs: list[plt.Axes]):
    """Scatterplot of UMAP visualization weighted by condition"""
    for i, cond in enumerate(conds):
        condList = np.where(np.asarray(totalconds == cond), cond, " Other Conditions")
        umap.plot.points(
            points,
            labels=condList,
            ax=axs[i],
            color_key_cmap="tab20",
            show_legend=True,
        )
        axs[i].set(
            title=decomp + "-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
        )


def plotGeneUMAP(
    genes: list[str],
    decomp,
    points: umap.UMAP,
    dataDF: pd.DataFrame,
    axs: list[plt.Axes],
):
    """Scatterplot of UMAP visualization weighted by gene"""
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        geneList = geneList / np.max(np.abs(geneList))
        psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
        plot = umap.plot.points(points, values=geneList, cmap=cmap, ax=axs[i])
        colorbar = plt.colorbar(psm, ax=plot)
        axs[i].set(
            title=genez + "-" + decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
        )


def plotCmpUMAP(
    cmp: int, factors: np.ndarray, pf2Points: umap.UMAP, allP: np.ndarray, ax: plt.Axes
):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = (allP @ factors)[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs)) * 2.0

    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)
    plot = umap.plot.points(pf2Points, values=weightedProjs, cmap=cmap, ax=ax)

    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    plt.colorbar(psm, ax=plot, label="Cell Specific Weight")

    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp))


def plotUMAP_obslabel(labels, pf2Points, ax: plt.Axes):
    """Scatterplot of UMAP visualization labeled by cell type or other obs column"""
    umap.plot.points(pf2Points, labels=labels, color_key_cmap="Paired", ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label " + str(labels.name),
    )


def plotLabelAllUMAP(conditions, points, ax: plt.Axes):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    umap.plot.points(
        points, labels=conditions, ax=ax, color_key_cmap="tab20", show_legend=True
    )
    ax.set(title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1")


def plotCellTypeUMAP(points, data, ax):
    """Plots UMAP labeled by cell type"""
    umap.plot.points(points, labels=data["Cell Type"].values, ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1")


def plotCmpPerCellType(weightedprojs, cmp, ax: plt.Axes, outliers=True):
    """Boxplot of weighted projections for one component across cell types"""
    cmpName = "Cmp. " + str(cmp)
    sns.boxplot(
        data=weightedprojs[[cmpName, "Cell Type"]],
        x=cmpName,
        y="Cell Type",
        showfliers=outliers,
        ax=ax,
    )
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(xlim=(-maxvalue, maxvalue), xlabel="Cell Specific Weight")
    ax.set_title(cmpName)
