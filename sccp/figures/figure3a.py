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
    ax, f = getSetup((8, 10), (5, 6))
    subplotLabel(ax) # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )
    allP = np.concatenate(projs, axis=0)
    CompW = allP @ factors[1]

    # UMAP dimension reduction
    ump = umap.UMAP(random_state=1).fit_transform(allP)
    x = ump[::10, 0]
    y = ump[::10, 1]

    for i in range(rank):
        tl = ax[i].scatter(
            x,
            y,
            marker='.',
            c=CompW[::10, i],
            cmap="PRGn",
            vmin=-0.3,
            vmax=0.3,
            linewidths=0,
            s=2.0,
        )
        # f.colorbar(tl, ax=ax[i])
        ax[i].set(title=f"Component {i + 1}")
        umap_axis(x, y, ax[i])

    return f
