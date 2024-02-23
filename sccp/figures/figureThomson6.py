"""
Thomson: Gene ontology for gene factors of Pf2
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology
from anndata import read_h5ad


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    geneOntology(X, 20)

    return f
