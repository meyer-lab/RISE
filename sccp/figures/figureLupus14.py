"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import getSetup, openPf2, openUMAP
from .commonFuncs.plotUMAP import plotCmpUMAP
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((28, 14), (5, 8))

    rank = 40
    _, factors, projs = openPf2(rank, "lupus", optProjs=True)

    # UMAP dimension reduction
    pf2Points = openUMAP(rank, "lupus", opt=True)

    component = np.arange(1, rank + 1, 1)

    for i in np.arange(0, rank):
        plotCmpUMAP(component[i], factors[1], pf2Points, projs, ax[i])

    return f
