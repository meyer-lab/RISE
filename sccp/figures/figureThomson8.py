"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import subplotLabel, getSetup, flattenData
from .commonFuncs.plotGeneral import plotGenePerCategStatus, plotGeneFactors
from ..imports.scRNA import ThompsonXA_SCGenes
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 24), (10, 6))

    rank = 30

    for i in np.arange(0, rank):
        plotGeneFactors(i + 1, rank, "Thomson", ax[2*i: 2*i + 2], geneAmount=5)

    return f
