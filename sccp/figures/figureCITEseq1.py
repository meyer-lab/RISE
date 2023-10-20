"""
CITEseq: Plotting Pf2 factors, weights, and UMAP labeled by all conditions
"""
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import plotFactors, plotWeight
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import plotAllLabelsUMAP
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = import_citeseq()
    X = pf2(X, "Condition", rank=40)

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(factors, X, ax[0:3], reorder=(0, 2), trim=(2,))
    plotWeight(X.uns["Pf2_weights"], ax[3])

    plotAllLabelsUMAP(X, "Condition", ax[4])

    return f
