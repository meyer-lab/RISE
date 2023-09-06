"""Gene ontology for gene factors of Pf2"""
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneOntology import (
    plotCombGO,
    plotPvalGO
)
from ..geneontology import geneOntology


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # value = "Underexpressed"
    value = "Overexpressed"

    # combDF, pvalDF = geneOntology(cmpNumb=26, geneAmount=30, goTerms=5, geneValue=value)

    # plotCombGO(combDF, geneValue=value, axs=ax[0:3])
    # plotPvalGO(pvalDF, geneValue=value, axs=ax[3:6])

    return f
