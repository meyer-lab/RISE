"""
Thomson: Further examination of cells based on their components
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotGeneral import cell_comp_hist
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import data
    rank = 30
    X = openPf2(rank, "Thomson")
    gateThomsonCells(X)
    cell_comp_hist(X, "Condition", 12, "Triamcinolone Acetonide", ax[0])
    cell_comp_hist(X, "Cell Type", 12, unique=None, ax=ax[1])
    print(X.obs["Cell Type"])

    cell_comp_hist(X, "Condition", 30, "Triamcinolone Acetonide", ax[2])
    cell_comp_hist(X, "Cell Type", 30, unique=None, ax=ax[3])


    return f
