"""
Lupus: Gene ontology for gene factors of Pf2
"""
import anndata
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

    X = anndata.read_h5ad(f"/opt/pf2/Lupus_analyzed_40comps.h5ad", backed="r")

    df = geneOntology(X, 32)

    print(df)

    return f
