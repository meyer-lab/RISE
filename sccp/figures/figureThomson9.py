"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import getSetup, openPf2, flattenWeightedProjs
from .commonFuncs.plotUMAP import plotCmpPerCellType
import numpy as np
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 20), (5, 6))

    rank = 30
    _, factors, projs = openPf2(rank, "Thomson")
    data = ThompsonXA_SCGenes()
    
    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = gateThomsonCells()
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    component = np.arange(1, rank + 1, 1)
    for i, comp in enumerate(component):
        plotCmpPerCellType(weightedProjDF, comp, ax[i], outliers=False)
        
        
    return f
