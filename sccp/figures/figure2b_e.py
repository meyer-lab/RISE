"""
Figure 2b_e
"""

import pacmap
import numpy as np
import anndata
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_labels_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")

    drug = ["Alprostadil"]
    plot_labels_pacmap(X, "Condition", ax[0], drug, cmap="Set1")
    ax[0].set(title="Pf2-Based Decomposition")

    plot_labels_pacmap(X, "Cell Type", ax[2])

    pc = PCA(n_components=20)
    pcaPoints = pc.fit_transform(np.asarray(X.X - X.var["means"].values))
    X.obsm["X_pf2_PaCMAP"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    plot_labels_pacmap(X, "Condition", ax[1], drug, cmap="Set1")
    ax[1].set(title="PCA-Based Decomposition")

    plot_labels_pacmap(X, "Cell Type", ax[3])

    return f
