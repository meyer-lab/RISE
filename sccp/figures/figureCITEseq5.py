"""
CITEseq: Average gene expression stratified by Leiden cluster and condition
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_celltype


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 42), (20, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    comps = [22, 33, 47, 48, 23, 31, 43]
    genes = bot_top_genes(X, cmp=comps[6], geneAmount=10)

    for i, gene in enumerate(genes):
        plot_avegene_per_celltype(X, gene, ax[i], cellType="leiden")
        ax[i].get_legend().remove()

    return f
