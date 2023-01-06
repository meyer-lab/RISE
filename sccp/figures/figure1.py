"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.scRNA import ThompsonXA_SCGenes
from tensorly.decomposition import parafac2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Gene, Cell, Drug]
    drugXA = ThompsonXA_SCGenes()

    # Performing parafac2 on single-cell Xarray
    rank = 5
    weights, factors, _ = parafac2(
        drugXA.to_numpy(),
        rank=rank,
        tol=1e-8,
        nn_modes=(0, 2),
        normalize_factors=True,
        verbose=True
    )
    
    plotSCCP_factors(rank, factors, drugXA, ax)
    
    return f
