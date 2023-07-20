"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    openPf2
)
from ..imports.scRNA import ThompsonXA_SCGenes
import umap
import umap.plot
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    rank = 30
    data = ThompsonXA_SCGenes(offset=1.0)
    
    weight, factors, projs = openPf2(rank, "Thomson")
    
    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(projs)
    
    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    subset = np.random.choice(a=[False, True], size=np.shape(projs)[0], p=[.93, .07])

    umap.plot.points(pf2Points,theme='blue', subset_points= subset, ax=ax[0])
    ax[0].set(
            title="Pf2-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1")
    
    
    umap.plot.points(pcaPoints,theme='red', subset_points=subset, ax=ax[1])
    ax[1].set(
            title="PCA-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1")
    
    return f
