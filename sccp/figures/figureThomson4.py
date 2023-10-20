"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotGeneUMAP, plotCondUMAP
import pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, dataName="Thomson")

    genes = ["GNLY", "NKG7"]
    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "Pf2", X, ax[i])

    drugs = ["Triamcinolone Acetonide", "Alprostadil"]
    for i, drug in enumerate(drugs):
        plotCondUMAP(drug, "Pf2", X, ax[i + 2])

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(X.X)
    X.obsm["embedding"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "PCA", X, ax[i + 4])

    for i, drug in enumerate(drugs):
        plotCondUMAP(drug, "PCA", X, ax[i + 6])

    return f
