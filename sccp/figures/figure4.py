"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, 
getSetup, plotFactors, 
plotR2X, plotCV, plotWeight,
openPf2, savePf2)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30

    # weight, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )
    
    # savePf2(weight, factors, projs, "Thomson")
    weight, factors, projs = openPf2(rank, "Thomson")

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    plotWeight(weight, ax[3])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f

