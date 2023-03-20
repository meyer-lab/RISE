"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import xarray as xa
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from sklearn import preprocessing
from matplotlib import gridspec, pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_samples

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


def plotFactorsSynthetic(factors, data_xarray, ax):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in range(0, len(factors)):
        if i != len(factors) - 2:
            if i == 0:
                timeDF = pd.DataFrame(factors[i], columns=xticks)
                timeDF["Time"] = np.arange(1, data_xarray.shape[i] + 1)
                sns.lineplot(data=timeDF[xticks], ax=ax[iter])
                ax[iter].set(
                    ylabel="Cmp. Weight",
                    xlabel="Time",
                    xticks=np.arange(0, data_xarray.shape[i]),
                )

            else:
                yt = data_xarray.coords[data_xarray.dims[i]].values
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


def plotFactors(factors, data_xarray, axs, reorder=tuple(), trim=tuple()):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in range(0, len(factors)):
        # The single cell mode has a square factors matrix
        if i != len(factors) - 2:
            yt = data_xarray.coords[data_xarray.dims[i]].values
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
                    print("Bottom 10 Genes Cmp." + str(j + 1) + ":", sort_data[:10])
                    print(
                        "Top 10 Genes Cmp." + str(j + 1) + ":", np.flip(sort_data[-10:])
                    )


def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def plotSS(projs: xa.Dataset, ax: matplotlib.axes._axes.Axes):
    proj_data = projs["projections"].to_numpy()
    assert proj_data.ndim == 2
    celltypeDF = projs["Cell Type"].to_dataframe()

    le = preprocessing.LabelEncoder()
    celltypes = le.fit_transform(celltypeDF["Cell Type"])

    df = pd.DataFrame([])
    for k in range(proj_data.shape[0]):
        ss = silhouette_samples(proj_data[k, :].reshape(-1, 1), celltypes)
        ss_scores = [np.mean(ss[celltypes == l]) for l in range(len(le.classes_))]

        ddf = pd.DataFrame(
            {
                "Silhoutte Score": ss_scores,
                "Cell Type": le.classes_,
                "Cmp.": [f"Cmp. {k+1}"] * len(le.classes_),
            }
        )
        df = pd.concat([df, ddf])

    sns.barplot(data=df, x="Cell Type", y="Silhoutte Score", hue="Cmp.", ax=ax)
    ax.tick_params(axis="x", rotation=45)


def plotProj(projs, axs):
    """Plot a projection matrix along with cell type annotations."""
    celltypeDF = projs["Cell Type"].to_dataframe()
    pjArr = projs["projections"].to_numpy().T

    le = preprocessing.LabelEncoder()
    celltypes = le.fit_transform(celltypeDF["Cell Type"])
    celltypesName = np.unique(celltypeDF["Cell Type"])

    idxx = np.argsort(celltypes)
    gini_index = giniIndex(pjArr)
    
    pjArr = pjArr[idxx, :]
    xticks = projs["projections"].coords["Cmp"].values
    
    sns.heatmap(
        data=np.flip(pjArr[:, gini_index],axis=0),
        xticklabels=xticks[gini_index],
        yticklabels=False,
        center=0,
        ax=axs[0],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
    )

    sns.heatmap(
        data=np.flip(celltypes[idxx].reshape((-1, 1))),
        xticklabels=False,
        yticklabels=False,
        ax=axs[1],
        cmap=sns.color_palette("tab10", len(celltypesName)),
    )

    colorbar_numbers = np.arange(0, len(celltypesName))
    cbar = axs[1].collections[0].colorbar
    cbar.set_ticks(colorbar_numbers)
    cbar.set_ticklabels(celltypesName)
    
    
def giniIndex(proj_data):
    """Calculates the Gini Coeff for each component and saves index rearrangment"""
    gini = np.empty(proj_data.shape[1])
    for i in range(proj_data.shape[1]):
        projComp = np.sort(proj_data[:, i])
        if np.amin(projComp) < 0:
            projComp -= np.amin(projComp)
            
        index = np.arange(1, projComp.shape[0]+1)
        gini[i] = (np.sum((2 * index - projComp.shape[0]  - 1) * projComp)) / (projComp.shape[0] * np.sum(projComp))
   
    giniIndex = np.argsort(gini)
    
    return giniIndex
