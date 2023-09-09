"""
Hamad CITEseq dataset
"""
import numpy as np
import umap
from parafac2 import parafac2_nd

from .common import (
    subplotLabel,
    getSetup,
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
    data = import_citeseq()
    rank = 40

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    component = np.arange(1, 25, 1)

    for i in range(len(component)):
        plotCmpUMAP(
            component[i], factors, pf2Points, np.concatenate(projs, axis=0), ax[i]
        )

    return f
