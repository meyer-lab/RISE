"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA = ThompsonXA_SCGenes(saveXA=False)

    # Performing parafac2 on single-cell Xarray
    _, factors, projs = parafac2(
        drugXA.to_numpy(),
        rank=5,
        verbose=False,
    )

    plotSCCP_factors(factors, drugXA, projs[:3, :, :], ax)

    return f
