"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 18), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    X = openPf2(40, "Lupus")

    comps = [13, 14, 16, 26, 29, 32]
    for i, cmp in enumerate(comps):
        plotCmpPerCellType(X, cmp, ax[(2 * i)])
        plotCmpUMAP(X, cmp, ax[(2 * i) + 1])

    return f
