"""
Thomson: Gene ontology for gene factors of Pf2
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    df = geneOntology(X, 20)

    print(df)

    return f
