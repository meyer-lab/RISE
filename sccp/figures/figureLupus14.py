"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import getSetup, openPf2, flattenWeightedProjs
from .commonFuncs.plotUMAP import plotCmpPerCellType
import numpy as np
from ..imports.scRNA import load_lupus_data

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 28), (8, 5))

    rank = 40
    lupus_tensor, obs = load_lupus_data()
    _, factors, projs = openPf2(rank, "lupus", optProjs=True)

     # get cell types
    cell_types = obs[["cell_type_broad", "SLE_status"]].reset_index(drop=True)

    weightedProjDF = flattenWeightedProjs(lupus_tensor, factors[1], projs)
    weightedProjDF["Cell Type"] = cell_types["cell_type_broad"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    component = np.arange(1, rank + 1, 1)
    for i, comp in enumerate(component):
        plotCmpPerCellType(weightedProjDF, comp, ax[i], outliers=False)
        

    return f
