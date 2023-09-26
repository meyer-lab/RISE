"""
Determining differces in raw gene expression for lupus status
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, flattenData, repeatLabels
from ..imports.scRNA import load_lupus_data
from .commonFuncs.plotGeneral import plotGenePerCategStatus
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, annData = load_lupus_data()
    dataDF = flattenData(data)
    
    condLabels = annData[["sample_ID", "SLE_status"]].drop_duplicates("sample_ID")
    condLabels = np.asarray(condLabels["SLE_status"])
 
    dataDF["Status"] = repeatLabels(condLabels, data, dataDF)
    
    cell_types = annData[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values
    
    comp = 26
    rank = 40
    plotGenePerCategStatus(comp, dataDF, rank, "lupus", ax[0:4], geneAmount=2)

    return f