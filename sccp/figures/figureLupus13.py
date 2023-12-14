"""
Lupus: Plots all top and bottom genes Pf2 weights
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (10, 8))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")
    comps = [13, 15, 19, 38]
    for i, cmp in enumerate(comps):
        plotGeneFactors(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plotGeneFactors(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
