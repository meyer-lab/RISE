"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import umap
from .common import subplotLabel, getSetup, umap_axis
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((4, 6), (2, 2))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30
    _, factors, _, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    # UMAP dimension reduction
    umapp = umap.UMAP(random_state=1, n_neighbors=5)
    ump = umapp.fit_transform(factors[2])
    x = ump[:, 0]
    y = ump[:, 1]

    ax[0].scatter(
        x,
        y,
        c="k",
        marker=".",
        linewidths=0,
        s=2.0,
    )
    ax[0].set(title=f"Genes")
    umap_axis(x, y, ax[0])

    return f
