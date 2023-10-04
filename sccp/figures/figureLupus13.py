"""
Plots all top and bottom genes for Lupus dataset
"""
from .common import getSetup
from .commonFuncs.plotGeneral import  plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (10, 8))

    rank = 40

    for i in np.arange(0, rank):
        plotGeneFactors(i + 1, rank, "lupus", ax[2*i: 2*i + 2], geneAmount=5)

    return f
