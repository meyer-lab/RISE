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
import pandas as pd
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
    rank = 24

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
        rank=rank,
        random_state=42
    )

    sort_idx = np.argsort(factors[2], axis=0)
    geneDF = pd.DataFrame()
    for j in range(rank):
        sort_data = X.variable_labels[sort_idx[:, j]]
        geneDF[str(j+1)] = np.ravel([sort_data[:10], np.flip(sort_data[-10:])])
    geneDF.to_csv("Perturb_Comps.csv")

    sort_idx = np.argsort(np.abs(factors[0]), axis=0)
    CRISPR_DF = pd.DataFrame()
    for j in range(rank):
        sort_data = X.condition_labels[sort_idx[:, j]]
        CRISPR_DF[str(j+1)] = np.flip(sort_data[-10:])
        print(np.abs(factors[0])[sort_idx[:, j]])
    #CRISPR_DF.to_csv("Perturb_Comps_CR.csv")

    plotFactors(factors, X, ax[0:3], reorder=(0, 2), trim=(2,))

    sns.heatmap(
        data=reorder_table(projs[0])[0],
        center=0,
        ax=ax[3],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
    )

    # plotR2X(X, 24, ax[3])

    return f
