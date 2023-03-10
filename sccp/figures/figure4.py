"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotFactors, plotProjs_SS, renamePlotscRNA, plotAllProjs_SS
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    drugXA, celltypeXA = ThompsonXA_SCGenes(saveXA=False, offset=1.0)

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _, _ = parafac2_nd(
        drugXA.to_numpy(),
        rank=6,
        verbose=True,
    )
    
    plotFactors(factors, drugXA, ax, reorder=(0, 1), trim=(2,))
    plotProjs_SS(factors, projs[3:5, :, :], celltypeXA, color_palette, ax)
    renamePlotscRNA(ax)

    plotR2X_CC(drugXA.to_numpy(), 8, ax[8], ax[9])
    
    plotAllProjs_SS(factors, projs, celltypeXA, color_palette, ax[10], ax[11], ax[12], size=20)
    
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
