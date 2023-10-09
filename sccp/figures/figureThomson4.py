"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup, openPf2, flattenData
from .commonFuncs.plotUMAP import plotGeneUMAP, points
from ..imports.scRNA import ThompsonXA_SCGenes
import pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30
    dataDF = flattenData(data)

    # UMAP dimension reduction
    _, _, projs = openPf2(rank, "Thomson")
    pf2Points = pacmap.PaCMAP().fit_transform(projs)

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = pacmap.PaCMAP().fit_transform(pcaPoints)

    genes = ["GNLY", "NKG7"]
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:2])
    plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # Find cells associated with drugs
    drugs = [
        "Triamcinolone Acetonide",
        "Alprostadil",
    ]
    condList = np.array([c if c in drugs else " Other Conditions" for c in dataDF["Condition"].values])

    points(
        pf2Points,
        labels=condList,
        ax=ax[4],
        color_key_cmap="Paired",
        show_legend=True,
    )
    ax[4].set(
        title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
    )

    points(
        pcaPoints,
        labels=condList,
        ax=ax[5],
        color_key_cmap="Paired",
        show_legend=True,
    )
    ax[5].set(
        title="PCA-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
    )

    return f
