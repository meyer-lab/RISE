"""
Lupus: Plots percentages of cell types in weighted proejctions above a threshold for a component
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import investigate_comp


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 2))  

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank=rank, dataName="Lupus")
    
    component = 13
    investigate_comp(X, component, "Cell Type", ax[0], threshold=0.1)
    investigate_comp(X, component, "Cell Type", ax[1], threshold=-0.1)

    return f
