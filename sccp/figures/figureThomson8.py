"""
Thomson: Plots all top and bottom genes Pf2 weights
"""
from .common import getSetup, openPf2
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 24), (10, 6))

    rank = 30
    X = openPf2(rank, "Thomson")

    for i in np.arange(0, rank):
        plotGeneFactors(i + 1, X, ax[2 * i : 2 * i + 2], geneAmount=5)

    return f
