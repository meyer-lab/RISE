"""
Hamad CITEseq dataset
"""
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotWeight,
)
from ..imports.citeseq import import_citeseq, combine_all_citeseq
from parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # combine_all_citeseq(saveAdata=True)

    data = import_citeseq()
    rank = 2

    weight, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    plotWeight(weight, ax[3])

    return f
