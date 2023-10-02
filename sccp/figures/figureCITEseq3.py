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
from .commonFuncs.plotUMAP import plotGeneUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    data, protDF = import_citeseq()
    rank = 80

    _, _, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    protNames = np.unique(protDF.drop(columns="Condition").columns)

    # protNames = protNames[0:24]
    # protNames = protNames[24:48]
    # protNames = protNames[50:75]
    # protNames = protNames[75:100]
    protNames = protNames[100:]

    plotGeneUMAP(protNames, "Pf2", pf2Points, protDF, ax[0:25])

    return f
