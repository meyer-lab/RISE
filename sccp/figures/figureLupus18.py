"""
Lupus: UMAP labeled by cell type
"""

from anndata import read_h5ad
import pandas as pd
import seaborn as sns
import numpy as np
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotLabelsUMAP
from ..gating import getHiResOldLupus


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")
    X = getHiResOldLupus(X)
    X.obs["Cell Type2"] = X.obs["Cell Type2"].cat.set_categories(
        np.sort(X.obs["Cell Type2"].cat.categories.values)
    )
    popCrossTabPlot(X, "Cell Type2", "Cell Type Old2", ax[2])

    plotLabelsUMAP(X, "Cell Type2", ax[0])
    plotLabelsUMAP(X, "Cell Type Old2", ax[1])

    return f


def popCrossTabPlot(X, column1, column2, ax):
    """Makes confsion matrix for old and new cell type labels"""
    conf = pd.crosstab(X.obs[column1], X.obs[column2])
    conf = conf / conf.sum(axis=0)
    sns.heatmap(conf, ax=ax, vmin=0, vmax=1)
