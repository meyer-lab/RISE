"""
Lupus: UMAP labeled by cell type
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotLabelsUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")

    plotLabelsUMAP(X, "Cell Type", ax[0])

    return f
