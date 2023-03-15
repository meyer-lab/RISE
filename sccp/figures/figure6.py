"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    reorder_table,
    plotFactors,
)
from ..imports.scRNA import import_perturb_RPE
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
import seaborn as sns


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
    X = X[:1200, :, :]

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        X.to_numpy(),
        rank=24,
        verbose=True,
    )

    plotFactors(factors, X, ax[0:2], reorder=(0, 2), trim=(2,))

    sns.heatmap(
        data=reorder_table(projs[0, :, :])[0],
        center=0,
        ax=ax[2],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
    )

    plotR2X(X.to_numpy(), 6, ax[2])

    return f
