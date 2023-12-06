"""
Thomson: Further examination of cells based on their components
"""
import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import cell_comp_hist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import data
    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")
    #cell_comp_hist(X, "Condition", 26, "Dexrazoxane HCl (ICRF-187, ADR-529)", ax[0])
    cell_comp_hist(X, "Cell Type", 12, unique=None, ax=ax[1])

    #cell_comp_hist(X, "Condition", 30, "Triamcinolone Acetonide", ax[2])
    #cell_comp_hist(X, "Cell Type", 30, unique=None, ax=ax[3])

    return f
