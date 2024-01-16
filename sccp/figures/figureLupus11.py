"""
Lupus: Gene ontology for gene factors of Pf2
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    df = geneOntology(X, 21)

    # cmp = [28, 21, 10, 4, 2, 1]

    print(df["Term"].values)

    return f
