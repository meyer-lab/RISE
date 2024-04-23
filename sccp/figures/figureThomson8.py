"""
Thomson: Plots all top and bottom genes Pf2 weights
"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 24), (10, 4))

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=20, top=True)
        plot_gene_factors_partial(i + 1, X, ax[2 * i + 1], geneAmount=20, top=False)

    return f
