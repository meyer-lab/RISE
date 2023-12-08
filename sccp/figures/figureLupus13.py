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

    for i in np.arange(0, 3):
        plotGeneFactors(i + 1, X, ax[2 * i], geneAmount=5, top=True)
        plotGeneFactors(i + 1, X, ax[2 * i + 1], geneAmount=5, top=False)

    return f
