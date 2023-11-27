"""
Thomson: Boxplots of weighted projectoins per component
"""
import anndata
import numpy as np
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 14), (5, 6))

    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    gateThomsonCells(X)

    component = np.arange(1, X.uns["Pf2_A"].shape[1] + 1, 1)
    for i, comp in enumerate(component):
        plotCmpPerCellType(X, comp, ax[i], outliers=False)

    return f
