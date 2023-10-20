"""
Lupus: Plotting weighted projections per component
"""
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 28), (8, 5))

    X = openPf2(rank=40, dataName="Lupus")

    for i in range(3):
        plotCmpPerCellType(X, i + 1, ax[i], outliers=False)

    return f
