"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import umap
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup, openPf2, flattenData
from .commonFuncs.plotUMAP import plotGeneUMAP, plotCondUMAP
from ..imports.scRNA import ThompsonXA_SCGenes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 30
    dataDF = flattenData(data)

    # UMAP dimension reduction
    _, _, projs = openPf2(rank, "Thomson")
    pf2Points = umap.UMAP(random_state=1).fit(projs)

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
    plotCondUMAP(drugs, "Pf2", dataDF["Condition"].values, pf2Points, ax[4:6])
    plotCondUMAP(drugs, "PCA", dataDF["Condition"].values, pcaPoints, ax[6:8])

    return f
