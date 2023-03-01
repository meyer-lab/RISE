"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotFactors
from ..imports.scRNA import import_perturb_RPE
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 20), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    X = import_perturb_RPE()
    print(X.shape)

    # Only have memory for some genes
    X = X[:600, :, :]

    # Performing parafac2 on single-cell Xarray
    _, factors, _, _, _ = parafac2_nd(
        X.to_numpy(),
        rank=6,
        verbose=True,
    )

    plotFactors(factors, X, ax, reorder=(0, 2), trim=(0, 2))

    plotR2X_CC(X.to_numpy(), 6, ax[2], ax[3])

    return f
