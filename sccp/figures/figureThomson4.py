"""
Thomson: Compares PCA and Pf2 UMAP labeled by genes and drugs
"""
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotGeneUMAP, plotLabelsUMAP
import pacmap
import numpy as np


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
        plotLabelsUMAP(X, "Condition", ax[i + 2], drug, cmap="Set1")
        ax[i + 2].set(title=f"Pf2-Based Decomposition")

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(np.asarray(X.X.to_memory() - X.var["means"].values))
    X.obsm["embedding"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "PCA", X, ax[i + 4])

    for i, drug in enumerate(drugs):
        plotLabelsUMAP(X, "Condition", ax[i + 6], drug, cmap="Set1")
        ax[i + 6].set(title=f"PCA-Based Decomposition")

    return f
