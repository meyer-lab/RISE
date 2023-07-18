"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP,
    plotCellStateViolins,
    plotUMAP_ct
)
from ..imports.scRNA import load_lupus_data
from ..parafac2 import parafac2_nd
import numpy as np
import umap
from sklearn.decomposition import PCA

 
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, cell_types, _ = load_lupus_data() # don't need to get patient color mappings
    rank = 30
    cellState = 28; cmp = 28

    # run pf2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)


    plotUMAP_ct(cell_types, pf2Points, projs, ax[0])
    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[1])
    plotCellStateViolins(projs, cell_types, cellState, ax[2])


    return f