"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    reorder_table,
    plotFactors,
)
import numpy as np
from ..imports.scRNA import import_perturb_RPE
from ..parafac2 import parafac2_nd
import seaborn as sns
import mygene


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 20), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    X = import_perturb_RPE()

    # Filter out sgRNAs with few cells
    delidx = np.array([xx.shape[0] > 200 for xx in X.X_list], dtype=bool)
    X.X_list = [X.X_list[ii] for ii in range(delidx.size) if delidx[ii]]
    X.condition_labels = X.condition_labels[delidx]
    X.condition_labels = np.array(
        [X.condition_labels[ii].split("_")[0] for ii in range(X.condition_labels.size)]
    )

    mg = mygene.MyGeneInfo()
    ginfo = mg.querymany(X.variable_labels, scopes="ensembl.gene")
    for ii in range(X.variable_labels.size):
        if "symbol" in ginfo[ii]:
            X.variable_labels[ii] = ginfo[ii]["symbol"]

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        X,
        rank=24,
        random_state=1,
    )

    plotFactors(factors, X, ax[0:2], reorder=(0, 2), trim=(2,))

    sns.heatmap(
        data=reorder_table(projs[0])[0],
        center=0,
        ax=ax[2],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
    )

    # plotR2X(X, 24, ax[3])

    return f
