"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from ..parafac2 import Pf2X


matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"


def getSetup(figsize, gridd, multz=None, empts=None, constrained_layout=True):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_lowercase[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from sccp.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def plotFactorsSynthetic(factors, data_xarray: Pf2X, ax):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in range(0, len(factors)):
        if i != len(factors) - 2:
            if i == 0:
                timeDF = pd.DataFrame(factors[i], columns=xticks)
                timeDF["Time"] = np.arange(1, factors[0].shape[0] + 1)
                sns.lineplot(data=timeDF[xticks], ax=ax[iter])
                ax[iter].set(
                    ylabel="Cmp. Weight",
                    xlabel="Time",
                    xticks=np.arange(0, factors[0].shape[0]),
                )

            else:
                yt = data_xarray.variable_labels
                X = factors[i]
                sns.heatmap(
                    data=X,
                    xticklabels=xticks,
                    yticklabels=yt,
                    ax=ax[iter],
                    center=0,
                    cmap=cmap,
                )

            ax[iter].set_title("Factors")
            ax[iter].tick_params(axis="y", rotation=0)
            iter += 1


def plotFactors(factors, data: Pf2X, axs, reorder=tuple(), trim=tuple()):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in (0, 2):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = data.condition_labels
        else:
            yt = data.variable_labels

        X = factors[i]

        if i in trim:
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > 0.08
            X = X[kept_idxs]
            yt = yt[kept_idxs]

        if i in reorder:
            X, ind = reorder_table(X)
            yt = yt[ind]

        sns.heatmap(
            data=X,
            xticklabels=xticks,
            yticklabels=yt,
            ax=axs[iter],
            center=0,
            cmap=cmap,
        )

        axs[iter].set_title("Factors")
        axs[iter].tick_params(axis="y", rotation=0)
        iter += 1

        if i == 2 and len(yt) > 50:
            sort_idx = np.argsort(X, axis=0)
            for j in range(rank):
                sort_data = yt[sort_idx[:, j]]
                # print("Bottom 10 Genes Cmp." + str(j + 1) + ":", sort_data[:10])
                # print("Top 10 Genes Cmp." + str(j + 1) + ":", np.flip(sort_data[-10:]))


def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def plotProj(projs, axs):
    """Plot a projection matrix along with cell type annotations."""
    sns.heatmap(
        data=projs,
        xticklabels=[f"Cmp. {i}" for i in np.arange(1, projs.shape[1] + 1)],
        yticklabels=False,
        center=0,
        ax=axs[0],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
    )


def flattenData(data, factors, projs):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(factors[0].shape[0]):
        cellCount = np.append(cellCount, projs[i].shape[0])

    flatProjs = np.empty([int(np.sum(cellCount)), projs[0].shape[1]])
    flatData = np.empty([int(np.sum(cellCount)), len(data.variable_labels)])
    cellStart = [0]
    drugNames = []

    for i in range(factors[0].shape[0]):
        cellStart = np.append(cellStart, cellStart[i] + cellCount[i])
        flatProjs[int(cellStart[i]) : int(cellStart[i + 1])] = projs[i]
        flatData[int(cellStart[i]) : int(cellStart[i + 1])] = data.X_list[i]
        drugNames = np.append(
            drugNames, np.repeat(data.condition_labels[i], cellCount[i])
        )

    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    projDF = pd.DataFrame(data=flatProjs, columns=cmpNames)
    dataDF = pd.DataFrame(data=flatData, columns=data.variable_labels)
    projDF["Drug"] = drugNames
    dataDF["Drug"] = drugNames

    return dataDF, projDF

def plotGeneUMAP(genes, decomp, points, dataDF, f, axs):
    """Scatterplot of UMAP visualization weighted by gene"""
    umap1 = points[::20, 0]
    umap2 = points[::20, 1]
    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        cmap=plt.cm.get_cmap('plasma')
        tl = axs[i].scatter(
            umap1, umap2, c=geneList[::20], cmap=cmap.reversed(), s=1,
        )
        f.colorbar(tl, ax=axs[i])
        axs[i].set(
            title=genez + "-" + decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
            xticks=np.linspace(np.min(umap1), 
                         np.max(umap1),
                         num=5),
            yticks=np.linspace(np.min(umap2), 
                         np.max(umap2),
                         num=5)
            )
        axs[i].axes.xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])

    return

def plotDrugUMAP(drugs, decomp, totaldrugs, points, axs):
    """Scatterplot of UMAP visualization weighted by condition"""
    umap1 = points[::20, 0]
    umap2 = points[::20, 1]
    for i, drugz in enumerate(drugs):
        drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Other Drugs")
        DF = pd.DataFrame(
            {
                "UMAP1": umap1,
                "UMAP2": umap2,
                "Drug": drugList[::20],
            }
        )
        sns.scatterplot(
            data=DF, x="UMAP1", y="UMAP2", hue="Drug", s=3, palette="muted", ax=axs[i]
        )
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles=handles, labels=labels)
        axs[i].set(
            title=decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
            xticks=np.linspace(np.min(umap1), 
                         np.max(umap1),
                         num=5),
            yticks=np.linspace(np.min(umap2), 
                         np.max(umap2),
                         num=5)
            )
        axs[i].axes.xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])

    return

def plotCmpUMAP(projDF, projName, points, f, axs):
    """Scatterplot of UMAP visualization weighted by projections for a component"""
    umap1 = points[::20, 0]
    umap2 = points[::20, 1]
    for i, proj in enumerate(projName):
        projs = projDF[proj].values
        cmap=plt.cm.get_cmap('plasma')
        tl = axs[i].scatter(
            umap1, umap2, c=projs[::20], cmap=cmap.reversed(), s=1,
        )
        f.colorbar(tl, ax=axs[i])
        axs[i].set(
            title=proj + "-Pf2-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
            xticks=np.linspace(np.min(umap1), 
                         np.max(umap1),
                         num=5),
            yticks=np.linspace(np.min(umap2), 
                         np.max(umap2),
                         num=5)
            )
        axs[i].axes.xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])
        
    return