"""
CITEseq: UMAP weighted by protein expression
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import plotGeneUMAP
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = import_citeseq()
    X = pf2(X, "Condition", rank=40)

    names = X.var_names[X.var["feature_types"] == "Antibody Capture"]

    protNames = names[0:25].tolist()

    for i, name in enumerate(protNames):
        plotGeneUMAP(name, "Pf2", X, ax[i])

    return f
