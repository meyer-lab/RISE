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
    plotUMAP_ct,
)
from ..imports.scRNA import load_lupus_data
from parafac2 import parafac2_nd
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
    data, obs = load_lupus_data(every_n=10)  # don't need to get patient color mappings
    rank = 5
    cmp = 4

    cell_types = obs["cell_type_broad"].reset_index(drop=True)

    # run pf2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        n_iter_max=10,
        random_state=1,
    )

    projs = np.concatenate(projs, axis=0)
    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1, verbose=True).fit(projs)

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    plotUMAP_ct(cell_types, pf2Points, ax[0])
    plotCmpUMAP(cmp, factors, pf2Points, projs, ax[1])

    return f
