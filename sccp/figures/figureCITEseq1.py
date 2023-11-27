"""
CITEseq: Plotting Pf2 factors, weights, and UMAP labeled by all conditions
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import plotFactors, plotWeight
from .commonFuncs.plotUMAP import plotLabelsUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")

    plotFactors(X, ax[0:3], reorder=(0, 2))
    plotWeight(X.uns["Pf2_weights"], ax[3])

    plotLabelsUMAP(X, "Condition", ax[4])

    return f
