"""
Determining differces in raw gene expression for lupus status
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, flattenData
from ..imports.scRNA import load_lupus_data
from .commonFuncs.plotGeneral import plotGenePerCategCond
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, annData = load_lupus_data()
    
    # condLabels = annData[["patient", "SLE_status"]].drop_duplicates("patient")
    # data.condition_labels = np.asarray(condLabels["SLE_status"])
    
    dataDF = flattenData(data)
    cell_types = annData[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values

    geneSet1 = ["HIST1H2AC", "PPBP", "CLU", "PF4", "TUBB1", "NRGN", "FCER1A", "CLEC10A", "HLA-DPA1", "HLA-DPB1", "HLA-DRB1", "HLA-DRA"]
    geneSet2 = ["IFI6", "MT2A", "LY6E", "ISG15", "TNFAIP3", "DUSP6", "ATP2B1", "KLF6", "SGK1", "EGR1"]
 
    # plotGenePerCategCond(geneSet1, dataDF, ax[0:13])
    # plotGenePerCategCond(geneSet2, dataDF, ax[13:24])

    return f