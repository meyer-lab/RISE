"""
Cytokine: Highly weighted genes per component
"""

from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial
from ..imports import import_cytokine
from ..factorization import correct_conditions
from ..factorization import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((21, 24), (10, 6))

    X = import_cytokine()
    X = pf2(X, 30, tolerance=1e-6)

    X.uns["Pf2_A"] = correct_conditions(X)

    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(i + 1, X, ax[(2 * i) + 1], geneAmount=10, top=False)

    return f
