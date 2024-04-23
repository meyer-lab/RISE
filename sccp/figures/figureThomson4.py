"""
Thomson: Compares PCA and Pf2 UMAP labeled by genes and drugs
"""

from anndata import read_h5ad
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plot_gene_pacmap, plot_labels_pacmap
import pacmap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad")

    genes = ["NKG7"]
    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, "Pf2", X, ax[i])

    drugs = ["Triamcinolone Acetonide", "Alprostadil", "Budesonide"]
    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 1], drug, cmap="Set1")
        ax[i + 2].set(title="Pf2-Based Decomposition")

    # PCA dimension reduction
    pc = PCA(n_components=30)
    pcaPoints = pc.fit_transform(np.asarray(X.X - X.var["means"].values))
    X.obsm["X_pf2_PaCMAP"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, "PCA", X, ax[i + 4])

    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 5], drug, cmap="Set1")
        ax[i + 6].set(title="PCA-Based Decomposition")

    return f
