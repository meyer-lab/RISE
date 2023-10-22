"""
CITEseq: UMAP weighted by protein expression
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import plotGeneUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = openPf2(80, "CITEseq")

    names = X.var_names[X.var["feature_types"] == "Antibody Capture"]

    protNames = names[0:4].tolist()

    for i, name in enumerate(protNames):
        plotGeneUMAP(name, "Pf2", X, ax[i])

    return f
