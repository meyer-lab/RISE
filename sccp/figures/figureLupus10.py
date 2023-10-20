"""
Lupus: Plot average gene expression of top/bottom genes for each component
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotGeneral import plotGenePerCategStatus


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    rank = 40
    X = openPf2(rank=rank, dataName="Lupus")
    cmp = 13
    plotGenePerCategStatus(X, cmp, rank, "Lupus", ax[0:4], geneAmount=2)

    return f
