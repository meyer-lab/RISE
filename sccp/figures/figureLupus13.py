"""
Lupus: Plots all top and bottom genes Pf2 weights
"""
from .common import getSetup, openPf2
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (10, 8))

    X = openPf2(rank=40, dataName="Lupus")

    for i in np.arange(0, 3):
        plotGeneFactors(i + 1, X, ax[2 * i : 2 * i + 2], geneAmount=5)

    return f
