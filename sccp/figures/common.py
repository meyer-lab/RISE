"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_letters
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from ..parafac2 import Pf2X
from ..crossVal import CrossVal
from ..decomposition import R2X
import os
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


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
            ascii_letters[ii],
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


def plotFactors(factors, data: Pf2X, axs, reorder=tuple(), trim=tuple()):
    """Plots parafac2 factors."""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    weight = 0.08
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = data.condition_labels
        elif i == 1:
            yt = [f"Cell State {i}" for i in np.arange(1, rank + 1)]
        else:
            yt = data.variable_labels

        X = factors[i]

        if i in trim:
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > weight
            X = X[kept_idxs]
            yt = yt[kept_idxs]

        if i in reorder:
            X, ind = reorder_table(X)
            yt = yt[ind]

        sns.heatmap(
            data=X,
            xticklabels=xticks,
            yticklabels=yt,
            ax=axs[i],
            center=0,
            cmap=cmap,
        )

        axs[i].set_title("Factors")
        axs[i].tick_params(axis="y", rotation=0)
        

        if i == 2 and len(yt) > 40:
            genesTop = np.empty((X.shape[0], X.shape[1]), dtype="<U10")
            genesBottom = np.empty((X.shape[0], X.shape[1]), dtype="<U10")
            sort_idx = np.argsort(X, axis=0)
            
            for j in range(rank):
                sortGenes = yt[sort_idx[:, j]]
                sortWeight= X[sort_idx[:, j], j] 
                genesIdxTop =  np.nonzero(sortWeight > 0.09)
                genesIdxBottom =  np.nonzero(sortWeight < -0.09)
                genesTop[:len(genesIdxTop[0]), j] = np.flip(sortGenes[genesIdxTop])
                genesBottom[:len(genesIdxBottom[0]), j] = sortGenes[genesIdxBottom]

            print(np.shape(genesTop))
            print(np.shape(genesBottom))
            dfTop = pd.DataFrame(data=genesTop, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])
            dfBottom = pd.DataFrame(data=genesTop, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])
            print(dfTop)
            print(dfBottom)
            
            dfTop.to_csv("TopGenes_Cmp"+str(rank)+".csv")
            dfBottom.to_csv("BottomGenes_Cmp"+str(rank)+".csv")
            # np.save(join(path_here, "sccp/data/TopGenes_Cmp"+str(rank)+".npy"), genesTop)
            # np.save(join(path_here, "sccp/data/BottomGenes_Cmp"+str(rank)+".npy"), genesBottom)
            
                


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

    drugNames = []

    for i in range(factors[0].shape[0]):
        drugNames = np.append(
            drugNames, np.repeat(data.condition_labels[i], cellCount[i])
        )

    flatProjs = np.concatenate(projs, axis=0)
    flatData = np.concatenate(data.X_list, axis=0)

    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    projDF = pd.DataFrame(data=flatProjs, columns=cmpNames)
    dataDF = pd.DataFrame(data=flatData, columns=data.variable_labels)
    weightedDF = pd.DataFrame(data=flatProjs @ factors[1], columns=cmpNames)
    projDF["Drug"] = drugNames
    dataDF["Drug"] = drugNames
    weightedDF["Drug"] = drugNames

    return dataDF, projDF, weightedDF


def plotGeneUMAP(genes, decomp, points, dataDF, f, axs):
    """Scatterplot of UMAP visualization weighted by gene"""
    umap1 = points[::10, 0]
    umap2 = points[::10, 1]
    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        cmap = plt.cm.get_cmap("plasma")
        tl = axs[i].scatter(
            umap1,
            umap2,
            c=geneList[::10],
            cmap=cmap.reversed(),
            s=0.1,
        )
        f.colorbar(tl, ax=axs[i])
        axs[i].set(
            title=genez + "-" + decomp + "-Based Decomposition",
        )
        umap_axis(umap1, umap2, axs[i])

    return


def plotDrugUMAP(drugs, decomp, totaldrugs, points, axs):
    """Scatterplot of UMAP visualization weighted by condition"""
    umap1 = points[::10, 0]
    umap2 = points[::10, 1]
    for i, drugz in enumerate(drugs):
        drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Other Drugs")
        sns.scatterplot(
            x=umap1, y=umap2, hue=drugList[::10], s=1, palette="muted", ax=axs[i]
        )
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles=handles, labels=labels)
        axs[i].set(
            title=decomp + "-Based Decomposition",
        )
        umap_axis(umap1, umap2, axs[i])

    return


def plotCmpUMAP(projDF, projName, points, f, axs):
    """Scatterplot of UMAP visualization weighted by projections for a component"""
    umap1 = points[::10, 0]
    umap2 = points[::10, 1]
    for i, proj in enumerate(projName):
        projs = projDF[proj].values
        cmap = plt.cm.get_cmap("plasma")
        tl = axs[i].scatter(
            umap1,
            umap2,
            c=projs[::10],
            cmap=cmap.reversed(),
            s=0.2,
        )
        f.colorbar(tl, ax=axs[i])
        axs[i].set(
            title=proj + "-Pf2-Based Decomposition",
        )
        umap_axis(umap1, umap2, axs[i])

    return


def umap_axis(x, y, ax):
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        xticks=np.linspace(np.min(x), np.max(x), num=5),
        yticks=np.linspace(np.min(y), np.max(y), num=5),
    )
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def plotR2X(data, rank, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = R2X(data, rank)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["Fit: Pf2", "Fit: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["|", "_"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            r2xError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )

    ax.legend()


def plotCV(data, rank, trainPerc, ax):
    """Creates variance explained plot for parafac2 tensor decomposition CV"""
    cvError = CrossVal(data, rank, trainPerc=trainPerc)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["CV: Pf2", "CV: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["o", "o"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            cvError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(0, np.max(np.append(cvError[0], cvError[1])) + 0.01, num=5),
    )

    ax.legend()
