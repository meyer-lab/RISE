"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAPDiv
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap 
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30
    
    weight, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )
    
    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit_transform(np.concatenate(projs, axis=0))
    
    cellState = np.arange(17, 25, 1) 
    component = np.arange(17, 25, 1) 
    
    for i in range(len(cellState)):
        plotCmpUMAPDiv(cellState[i], component[i], factors, pf2Points, projs, ax[i])

    return f
