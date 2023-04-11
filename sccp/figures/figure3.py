"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugDimReduc,
    plotGeneDimReduc,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (4, 3))

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

    # Mono1, Mono2, DC, NK, CD4, B, CD8]
    genes = ["CD14", "FCGR3A", "ITGAM", "NKG7", "IL7R", "MS4A1", "CD8A"]
    plotGeneDimReduc(genes, "Pf2", pf2Points, dataDF, f, ax[0:7])
    # plotGeneDimReduc(genes, "PCA", pcaPoints, dataDF, f, ax[0:7])

    # Find cells associated with drugs
    drugs = ["Triamcinolone Acetonide"]

    # Find cells associated with drugs
    drugs = [
        "Triamcinolone Acetonide",
        "Meprednisone",
        "Alprostadil",
        "Budesonide",
        "Betamethasone Valerate",
    ]
    plotDrugDimReduc(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[7:12])
    # plotDrugDimReduc(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[7:12])

    return f
