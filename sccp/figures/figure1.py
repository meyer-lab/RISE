"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.scRNA import ThompsonXA_SCGenes
from tensorly.decomposition import parafac2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA = ThompsonXA_SCGenes()

    # Performing parafac2 on single-cell Xarray
    _, factors, projs = parafac2(
        drugXA.to_numpy(),
        rank=5,
        n_iter_max=10,
        normalize_factors=True,
        verbose=True,
    )

    plotSCCP_factors(factors, drugXA, projs[0,:,:], ax)

    return f
