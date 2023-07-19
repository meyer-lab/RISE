"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA

from cProfile import Profile


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)


    pf = Profile()
    pf.enable()

    rank = 20
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        n_iter_max=20,
        verbose=True,
    )

    pf.disable()
    pf.dump_stats("profile")

    dataDF = flattenData(data)

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    genes = ["GNLY", "NKG7"]
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:2])
    plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # Find cells associated with drugs
    drugs = [
        "Triamcinolone Acetonide",
        "Alprostadil",
    ]
    plotDrugUMAP(drugs, "Pf2", dataDF["Condition"].values, pf2Points, ax[4:6])
    plotDrugUMAP(drugs, "PCA", dataDF["Condition"].values, pcaPoints, ax[6:8])
    
    cellState = 1; cmp = 1
    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[8])
    
    return f
