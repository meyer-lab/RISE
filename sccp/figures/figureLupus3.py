"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
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
        plotCmpPerCellType(X, cmp, ax[(2*i)])
        plotCmpUMAP(X, cmp, ax[(2*i) + 1])


    return f
