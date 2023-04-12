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
    plotCmpUMAP
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 30
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
    )
    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    pf2Points = umapReduc.fit_transform(projDF[cmpNames].to_numpy())

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(dataDF[data.variable_labels].to_numpy())
    pcaPoints = umapReduc.fit_transform(pcaPoints)
     
    # NK, CD4, B, CD8
    genes = ["NKG7", "IL7R", "MS4A1", "CD8A"]
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, f, ax[0:4])
    plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, f, ax[4:8])

    # Find cells associated with drugs
    drugs = [
        "Triamcinolone Acetonide",
        "Alprostadil",
    ]
    plotDrugUMAP(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[8:10])
    plotDrugUMAP(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[10:12])
    
    cmp = ["Cmp. 23", "Cmp. 25", "Cmp. 29", "Cmp. 30"]
    plotCmpUMAP(projDF, cmp, pf2Points, f, ax[12:16])
    
    return f


