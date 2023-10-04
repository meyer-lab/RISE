"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import getSetup, openPf2, openUMAP
from .commonFuncs.plotUMAP import plotCmpUMAP
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 14), (5, 6))

    rank = 30
    _, factors, projs = openPf2(rank, "Thomson")

    # UMAP dimension reduction
    pf2Points = openUMAP(rank, "Thomson", opt=False)

    component = np.arange(1, rank + 1, 1)

    for i in np.arange(0, rank):
        plotCmpUMAP(component[i], factors[1], pf2Points, projs, ax[i])

    return f
