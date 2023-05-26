"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import umap
import umap.plot as plt
from .common import subplotLabel, getSetup, umap_axis
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (3, 3))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )
    allP = np.concatenate(projs, axis=0)
    CompW = allP @ factors[1]

    # UMAP dimension reduction
    ump = umap.UMAP(random_state=1).fit(allP)

    for i in range(2):
        plt.points(
            ump, values=CompW[:, i], theme="fire", ax=ax[i], width=400, height=400
        )

        ax[i].set(title=f"Component {i + 1}")
        # umap_axis(x, y, ax[i])

    return f
