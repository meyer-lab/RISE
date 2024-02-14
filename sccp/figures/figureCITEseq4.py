"""
CITEseq: Plotting genes per component
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import plotGeneFactors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")
    comps = [22, 33, 47, 48, 23, 31, 43]

    for i, cmp in enumerate(comps):
        plotGeneFactors(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plotGeneFactors(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
