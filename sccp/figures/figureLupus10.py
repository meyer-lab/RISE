"""
Determining differces in raw gene expression for lupus status
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotGeneral import plotGenePerCategStatus
import numpy as np


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
    plotGenePerCategStatus(X,cmp, rank, "Lupus", ax[0:4], geneAmount=2)

    return f