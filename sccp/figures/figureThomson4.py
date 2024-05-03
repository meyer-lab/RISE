"""
Thomson: PCA and Pf2 PaCMAP labeled by genes and drugs
"""

from anndata import read_h5ad
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap
import pacmap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad")

    genes = ["NKG7"]
    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, "Pf2", X, ax[i])

    drugs = ["Triamcinolone Acetonide", "Alprostadil", "Budesonide"]
    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 1], drug, cmap="Set1")
        ax[i + 1].set(title="Pf2-Based Decomposition")

    plot_labels_pacmap(X, "Cell Type", ax[4])
    plot_labels_pacmap(X, "Cell Type2", ax[5])

    # PCA dimension reduction
    pc = PCA(n_components=20)
    pcaPoints = pc.fit_transform(np.asarray(X.X - X.var["means"].values))
    X.obsm["X_pf2_PaCMAP"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, "PCA", X, ax[i + 6])

    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 7], drug, cmap="Set1")
        ax[i + 7].set(title="PCA-Based Decomposition")

    plot_labels_pacmap(X, "Cell Type", ax[10])

    return f
