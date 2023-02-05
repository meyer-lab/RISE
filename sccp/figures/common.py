"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import sys
import time
import seaborn as sns
import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch

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


def plotSCCP_factors(factors, data_xarray, projs, ax, celltypeXA=None, color_palette=None, plot_celltype=False, reorder=tuple()):
    """Plots parafac2 factors and projection matrix"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for i in range(0, len(factors)):
        # The single cell mode has a square factors matrix
        if i == len(factors) - 2:
            yt = xticks
        else:
            yt = data_xarray.coords[data_xarray.dims[i]].values

        if i in reorder:
            reordered_projs, ind = reorder_table(factors[i])
            yt = yt[ind]
        else:
            reordered_projs = factors[i]
        
        if i == 2 and len(yt) > 20:
            min_idx = np.argsort(reordered_projs, axis=0)
            max_idx = np.argsort(reordered_projs, axis=0)
            for j in range(rank):
                print("Bottom 10 Genes Cmp." + str(j+1) + ":", yt[min_idx[:, j]])
                print("Top 10 Genes Cmp." + str(j+1) + ":", np.flip(yt[max_idx[:, j]]))   

        abs_value = np.max(np.abs([np.max(reordered_projs), np.min(reordered_projs)]))
        sns.heatmap(
            data=reordered_projs,
            xticklabels=xticks,
            yticklabels=yt,
            ax=ax[i],
            cmap=cmap,vmin=-abs_value,vmax=abs_value
        )

        ax[i].set_title("Mean Factors")
        ax[i].tick_params(axis="y", rotation=0)


    for i, ps in enumerate(projs):
        pps = ps[~np.all(ps == 0, axis=1)]
        reordered_projs, ind = reorder_table(pps)
        abs_value = np.max(np.abs([np.max(reordered_projs), np.min(reordered_projs)]))
        sns.heatmap(
            data=reordered_projs,
            xticklabels=xticks,
            yticklabels=ind,
            ax=ax[2*i + len(factors)],
            vmin=-abs_value,vmax=abs_value,
            cmap=cmap,
        )

        if plot_celltype == True:
            true_celltypes = celltypeXA[i, ind].to_dataframe().reset_index().set_index("Cell Type")
            celltypesDF = true_celltypes.drop(columns=true_celltypes.columns)
            allcelltypes = celltypesDF.copy()
            celltypesDF["Cell Type"] = 0
            label_colorbar = []
            choose_color_palette = []
            colorbar_numbers = np.arange(0, len(np.unique(allcelltypes.index)))
            for j, label in enumerate(np.unique(allcelltypes.index)):
                celltypesDF[celltypesDF.index == label] = j
                choose_color_palette = np.append(choose_color_palette, color_palette[j])
                label_colorbar = np.append(label_colorbar, label) 

            sns.heatmap(
                data=celltypesDF.to_numpy(),
                xticklabels=False,
                yticklabels=False,
                ax=ax[2*i + len(factors) + 1],
                cmap=list(choose_color_palette),
                )
        
            cbar = ax[2*i + len(factors) + 1].collections[0].colorbar
            cbar.set_ticks(colorbar_numbers)
            cbar.set_ticklabels(np.unique(allcelltypes.index))

def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def renamePlotSynthetic(xarray, ax):
    ax[0].set_yticklabels([f"Time:{i}" for i in np.arange(1, xarray.shape[0] + 1)])
    ax[3].set_title("Projection Matrix - " + "Time:0")
    ax[5].set_title("Projection Matrix - " + "Time:5")

def renamePlotIL2(ax):
    ax[2].set_yticklabels([f"Time:{i}" for i in [1, 2, 4]])
    ax[5].set_title("Projection Matrix - " + "R38Q N-term-2")
    ax[7].set_title("Projection Matrix - " + "R38Q/H16N-2")
    
def renamePlotscRNA(ax):
    ax[3].set_title("Projection Matrix - " + "Acetylcysteine")
    ax[5].set_title("Projection Matrix - " + "Adapalene")