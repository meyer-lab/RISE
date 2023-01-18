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


def plotSCCP_factors(factors, data_xarray, celltypeXA, projs, ax, color_palette, reorder=tuple()):
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

        sns.heatmap(
            data=reordered_projs,
            xticklabels=xticks,
            yticklabels=False,
            ax=ax[i],
            cmap=cmap,
        )

        ax[i].set_title("Mean Factors")
        ax[i].tick_params(axis="y", rotation=0)

    for i, ps in enumerate(projs):
        reordered_projs, ind = reorder_table(ps)
        
        sns.heatmap(
            data=reordered_projs,
            xticklabels=xticks,
            yticklabels=False,
            ax=ax[2*i + len(factors)],
            cmap=cmap,
        )

        true_celltypes = celltypeXA[i, ind].to_dataframe().reset_index().drop(columns=["Cell","Time"]).set_index("Label")
        true_celltypes["Type"] = 0
        label_colorbar = []
        colorbar_numbers = np.arange(0, len(np.unique(celltypeXA)))
        for j, label in enumerate(np.unique(celltypeXA)):
            true_celltypes[true_celltypes.index == label] = j
            label_colorbar = np.append(label_colorbar, label)   
    
        sns.heatmap(
            data=true_celltypes.to_numpy(),
            xticklabels=False,
            yticklabels=False,
            ax=ax[2*i + len(factors) + 1],
            cmap=color_palette,
            )
        
        cbar = ax[2*i + len(factors) + 1].collections[0].colorbar
        cbar.set_ticks(colorbar_numbers)
        cbar.set_ticklabels(label_colorbar)



def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


