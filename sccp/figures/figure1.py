"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.scRNA import ThompsonXA_SCGenes
from parafac2 import parafac2 as pf2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA = ThompsonXA_SCGenes()
    
    # Performing parafac2 on single-cell Xarray
    rank = 5
    _, factors, _ = pf2(drugXA.to_numpy(), rank, n_iter_max=2000, 
                        tol=1e-8, nn_modes=None, verbose=False,
                        n_iter_parafac=5)
    
    plotSCCP_factors(rank, factors, drugXA, ax)
    
    return f
