"""
Hamad CITEseq dataset
"""
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
from .commonFuncs.plotFactors import (
    plotFactors,
)
from .commonFuncs.plotUMAP import (
    plotCondUMAP
)
import umap
from ..imports.citeseq import import_citeseq, combine_all_citeseq
from parafac2 import parafac2_nd
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # combine_all_citeseq(saveAdata=True)

    data = import_citeseq()
    rank = 40

    weight, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    
    )

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)


    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    dataDF = flattenData(data)
    cond = [
        "control",
        "sc_pod1",
        "sc_pod7",
        "ic_pod1",
        "ic_pod7"
    ]
    plotCondUMAP(cond, "Pf2", dataDF["Condition"].values, pf2Points, ax[3:9])

    return f
