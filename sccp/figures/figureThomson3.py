"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
import pacmap
from .common import subplotLabel, getSetup, openPf2, openPf2
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    _, factors, projs = openPf2(rank, "Thomson")

    # UMAP dimension reduction
    _, _, projs = openPf2(rank, "Thomson")
    pf2Points = pacmap.PaCMAP().fit_transform(projs)

    component = np.arange(17, 25, 1)

    for i in range(len(component)):
        plotCmpUMAP(component[i], factors[1], pf2Points, projs, ax[i])

    return f
