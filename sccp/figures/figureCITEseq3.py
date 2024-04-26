"""
CITEseq: PaCMAP labeled by gene/protein expression
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import plot_gene_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    names = X.var_names[X.var["feature_types"] == "Antibody Capture"]

    for i, name in enumerate(names[0:4]):
        plot_gene_pacmap(name, "Pf2", X, ax[i])

    return f
