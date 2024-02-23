"""
Thomson: Compares PCA and Pf2 UMAP labeled by genes and drugs
"""
from anndata import read_h5ad
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotGeneUMAP, plotLabelsUMAP
import pacmap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad")

    genes = ["GNLY", "NKG7"]
    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "Pf2", X, ax[i])

    drugs = ["Triamcinolone Acetonide", "Alprostadil"]
    for i, drug in enumerate(drugs):
        plotLabelsUMAP(X, "Condition", ax[i + 2], drug, cmap="Set1")
        ax[i + 2].set(title="Pf2-Based Decomposition")

    # PCA dimension reduction
    pc = PCA(n_components=30)
    pcaPoints = pc.fit_transform(np.asarray(X.X - X.var["means"].values))
    X.obsm["X_pf2_PaCMAP"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "PCA", X, ax[i + 4])

    for i, drug in enumerate(drugs):
        plotLabelsUMAP(X, "Condition", ax[i + 6], drug, cmap="Set1")
        ax[i + 6].set(title="PCA-Based Decomposition")

    return f
