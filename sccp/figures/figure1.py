"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 15), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA = ThompsonXA_SCGenes()

    # Performing parafac2 on single-cell Xarray
    _, factors, projs = parafac2_nd(
        drugXA.to_numpy(),
        rank=5,
    )

    plotSCCP_factors(factors, drugXA, projs[:3, :, :], ax, reorder=(0, 2))

    return f
