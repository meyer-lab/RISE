"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import sys
import time
import seaborn as sns
import matplotlib
import numpy as np
import tensorly as tl
from matplotlib import gridspec, pyplot as plt
from matplotlib.patches import Ellipse
from gmm.tensor import markerslist, tensorGMM


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
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

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
        ax.text(-0.2, 1.2, ascii_lowercase[ii], transform=ax.transAxes, fontweight="bold", va="top")


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from gmm.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def add_ellipse(
    timei: int,
    dosei: int,
    ligandi: int,
    fac: tensorGMM,
    marker1,
    marker2,
    n_cluster: int,
    ax,
    colorpal,
    datatype,
):
    """Adding necessary conditons to form elipse around clusters for data points based on factors with respect to sns."""
    if datatype == "IL2":
        markerVec = np.zeros(len(markerslist), dtype=bool)
        markerVec[markerslist.index(marker1)] = 1
        markerVec[markerslist.index(marker2)] = 1

        markerProj = np.zeros((2, len(markerslist)), dtype=bool)
        markerProj[0, markerslist.index(marker1)] = 1
        markerProj[1, markerslist.index(marker2)] = 1
    else:
        markerVec = np.ones(2, dtype=bool)
        markerProj = np.eye(2, dtype=bool)

    means = tl.cp_to_tensor(fac)
    means = means[:, markerVec, timei, dosei, ligandi]
    coVars = fac.get_covariances()
    coVars = np.squeeze(np.array(coVars[:, :, :, timei, dosei, ligandi]))

    if datatype == "IL2":
        if markerslist.index(marker2) < markerslist.index(marker1):
            means = np.fliplr(means)

    for i in range(n_cluster):
        cholCov = coVars[i, :, :]
        covar = cholCov @ cholCov.T
        covar = markerProj @ covar @ markerProj.T
        S, U = np.linalg.eigh(covar)
        S = np.sqrt(S)
        angle = np.angle(U[0, 0] + U[1, 0] * 1j, deg=True)

        elipse = Ellipse(
            xy=means[i],
            width=3 * S[0],
            height=3 * S[1],
            edgecolor=colorpal[i],
            fill=False,
            facecolor=None,
            angle=angle,
        )

        ax.add_artist(elipse)

    return
