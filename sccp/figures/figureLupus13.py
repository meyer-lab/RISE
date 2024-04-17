"""
Lupus: Plots all top and bottom genes Pf2 weights
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotGeneral import plotGeneFactors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    # ax, f = getSetup((12, 10), (3, 4))

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    # comps = [4, 7, 8, 10, 22, 24]
    
    
    ax, f = getSetup((14, 16), (10, 6))
    for i in range(X.uns["Pf2_A"].shape[1]):
    # for i, cmp in enumerate(comps):
        plotGeneFactors(i+1, X, ax[2*i], geneAmount=10, top=True)
        plotGeneFactors(i+1, X, ax[(2*i)+1], geneAmount=10, top=False)

  

    return f
