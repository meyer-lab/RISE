"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotSCCP_factors, renamePlotscRNA
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 20), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA, celltypeXA = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    
    # Performing parafac2 on single-cell Xarray
    rank=6
    _, factors, projs, _, _ = parafac2_nd(
        drugXA.to_numpy(),
        rank=rank,
        verbose=True, 
    )

    plotSCCP_factors(
        factors,
        drugXA,
        projs[:2, :, :],
        ax,
        celltypeXA[:2, :],
        color_palette,
        reorder=(0, 2),
        trim=(2, ),
    )
    
    renamePlotscRNA(ax)

    plotR2X_CC(drugXA.to_numpy(), rank, ax[10], ax[11])

    return f


color_palette = [
    "black",
    "lightcoral",
    "red",
    "darkorange",
    "yellow",
    "green",
    "turquoise",
    "blue",
    "blueviolet",
    "plum",
    "pink",
    "saddlebrown",
    "gold",
    "grey",
]
