"""
Hamad CITEseq dataset
"""
import numpy as np
import umap

from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)
    rank = 80

    _, factors, projs = openPf2(rank=rank, dataName="CITEseq")

    pf2Points = umap.UMAP(random_state=1).fit(projs)

    component = np.arange(1, 25, 1)

    for i in range(len(component)):
        plotCmpUMAP(
            component[i], factors[1], pf2Points, projs, ax[i]
        )

    return f
