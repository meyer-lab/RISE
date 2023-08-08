"""
Investigation of raw data for Thomson dataset
"""
from .common import subplotLabel, getSetup, flattenData, plotCellTypePerExpCount, plotCellTypePerExpPerc
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (5, 5))

    # Add subplot labels
    subplotLabel(ax)


    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    dataDF["Cell Type"] = gateThomsonCells(rank=30, saveCellTypes=False)
    
    
    for i, drug in enumerate(data.condition_labels):
        if i < 12:
            plotCellTypePerExpCount(dataDF.loc[dataDF["Condition"] == drug], drug, ax[2*i])
            plotCellTypePerExpPerc(dataDF.loc[dataDF["Condition"] == drug], drug, ax[1+(2*i)])

    return f

