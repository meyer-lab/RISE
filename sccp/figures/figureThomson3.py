"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common.common import (
    subplotLabel,
    getSetup,
)
from .common.plotUMAP import (
    plotCmpUMAP,
)
from .common.saveFiles import openPf2, openUMAP

import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    _, factors, projs = openPf2(rank, "Thomson")

    # UMAP dimension reduction
    pf2Points = openUMAP(rank, "Thomson", opt=False)

    component = np.arange(17, 25, 1)

    for i in range(len(component)):
        plotCmpUMAP(component[i], factors, pf2Points, projs, ax[i])

    return f
