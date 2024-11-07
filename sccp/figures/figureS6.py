"""
Figure S6
"""

import anndata

from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_labels_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (6, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    
    plot_labels_pacmap(X, "Cell Type", ax[0])
    plot_labels_pacmap(X, "Cell Type2", ax[1])
    
    ax[2].axis("off")
    ax[3].axis("off")

    for i in range(1, 21):
        plot_wp_pacmap(X, i, ax[i + 3], cbarMax=0.3)

    return f
