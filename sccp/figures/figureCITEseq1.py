"""
Hamad CITEseq dataset
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import (
    plotFactors,
)
import umap
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import points


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    data, protDF = import_citeseq()
    rank = 80

    _, factors, projs = openPf2(rank=rank, dataName="CITEseq")

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))

    pf2Points = umap.UMAP(random_state=1).fit(projs)

    points(
        pf2Points,
        labels=protDF["Condition"].values,
        ax=ax[3],
        show_legend=True,
    )

    return f
