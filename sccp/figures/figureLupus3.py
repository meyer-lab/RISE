"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, flattenWeightedProjs, openPf2, openUMAP, flattenData
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP, plotCellTypeUMAP
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 18), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    lupus_tensor, obs = load_lupus_data()
    rank = 40
    
    dataDF = flattenData(lupus_tensor)

    # get cell types
    cell_types = obs[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    (
        _,
        factors,
        projs,
    ) = openPf2(rank=rank, dataName="lupus", optProjs=True)
    
    pf2Points = openUMAP(rank, "lupus", opt=True)

    weightedProjDF = flattenWeightedProjs(lupus_tensor, factors, projs)
    weightedProjDF["Cell Type"] = cell_types["cell_type_broad"].values
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values

    comps = [13, 14, 16, 26, 29, 32]
    for i, comp in enumerate(comps):
        plotCmpPerCellType(weightedProjDF, comp, ax[(2*i)], outliers=False)
        plotCmpUMAP(comp, factors, pf2Points, projs, ax[(2*i)+1])
    

    plotCellTypeUMAP(pf2Points, dataDF, ax[14])


    return f
