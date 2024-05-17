"""
Lupus: AUC ROC curve for predicing only fourth batch and each batch
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import plot_roc_fourthbatch
from ..factorization import correct_conditions, pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    X = pf2(X, rank=int(30), doEmbedding=False)
    X.uns["Pf2_A"] = correct_conditions(X)

    plot_roc_fourthbatch(X, ax[0])

    return f
