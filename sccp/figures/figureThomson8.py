"""
Thomson: Plots all top and bottom genes Pf2 weights
"""
import anndata
from .common import getSetup
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 24), (10, 6))

    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    for i in np.arange(0, X.uns["Pf2_A"].shape[1]):
        plotGeneFactors(i + 1, X, ax[2 * i], geneAmount=5, top=True)
        plotGeneFactors(i + 1, X, ax[2 * i + 1], geneAmount=5, top=False)

    return f
