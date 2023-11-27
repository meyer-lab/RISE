"""
Lupus: Plotting weighted projections per component
"""
import anndata
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 28), (8, 5))

    X = anndata.read_h5ad(f"/opt/pf2/Lupus_analyzed_40comps.h5ad", backed="r")

    for i in range(3):
        plotCmpPerCellType(X, i + 1, ax[i], outliers=False)

    return f
