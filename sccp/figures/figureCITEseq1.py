"""
CITEseq: Plotting Pf2 factors, weights, and UMAP labeled by all conditions
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
    plotWeight,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")
    XX = read_h5ad("/opt/pf2/CITE_Neighbors.h5ad", backed="r")
    X.obs["leiden"] = XX.obs["leiden"]

    plotConditionsFactors(X, ax[0])
    plotCellState(X, ax[1])
    plotGeneFactors(X, ax[2])
    plotWeight(X, ax[3])

    plotLabelsUMAP(X, "Condition", ax[4])
    plotLabelsUMAP(X, "leiden", ax[5])
    plotRatio(X, ax[6], day7=False)
    plotRatio(X, ax[7], day7=True)

    return f


def plotRatio(X, ax, day7=True):
    """Plots ratio of condition factors for day 1 or 7"""
    p = np.unique(X.obs["Condition"])
    p = [p[2], p[1], p[0], p[3], p[4]]
    d = X.uns["Pf2_A"]
    X = np.array([d[2], d[1], d[0], d[3], d[4]])
    xticks = np.arange(1, np.shape(X)[1] + 1)

    if day7 is False:
        ratio = X[1, :] / X[-2, :]
        day = 7
        yticks = [0.45, 1, 1.55]

    else:
        ratio = X[0, :] / X[-1, :]
        day = 1
        yticks = [2, 1.5, 1, 0.5, 0]

    ax.plot(xticks, ratio)
    ax.set(xticks=np.arange(1, np.shape(X)[1] + 1, 2), yticks=yticks)
    ax.set_xlabel("Components")
    ax.set_ylabel(f"IC/SC Ratio Day {day}")
