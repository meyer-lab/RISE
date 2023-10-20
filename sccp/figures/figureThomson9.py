"""
Thomson: Boxplots of weighted projectoins per component
"""
import numpy as np
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 20), (5, 6))

    rank = 30
    X = openPf2(rank, "Thomson")

    X.obs["Cell Type"] = gateThomsonCells(X)

    component = np.arange(1, rank + 1, 1)
    for i, comp in enumerate(component):
        plotCmpPerCellType(X, comp, ax[i], outliers=False)

    return f
