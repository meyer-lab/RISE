"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    reorder_table,
    flattenData,
    plotDrugDimReduc,
    plotGeneDimReduc
)
import numpy as np
from ..imports.scRNA import import_perturb_RPE
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
import seaborn as sns
import mygene
from ..parafac2 import parafac2_nd
import umap 
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    X = import_perturb_RPE()

    # Filter out sgRNAs with few cells
    delidx = np.array([xx.shape[0] > 200 for xx in X.X_list], dtype=bool)
    X.X_list = [X.X_list[ii] for ii in range(delidx.size) if delidx[ii]]
    X.condition_labels = X.condition_labels[delidx]
    X.condition_labels = np.array([X.condition_labels[ii].split("_")[0] for ii in range(X.condition_labels.size)])

    mg = mygene.MyGeneInfo()
    ginfo = mg.querymany(X.variable_labels, scopes='ensembl.gene')
    for ii in range(X.variable_labels.size):
        if "symbol" in ginfo[ii]:
            X.variable_labels[ii] = ginfo[ii]["symbol"]

    data = X
    # Performing parafac2 on single-cell Xarray
    rank = 30
    _, factors, projs, _ = parafac2_nd(
        X,
        rank=24,
        verbose=True,
    )


    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    pf2Points = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
    
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(dataDF[data.variable_labels].to_numpy())
    pcaPoints = umapReduc.fit_transform(pcaPoints)

    # Mono1, Mono2, NK, CD4, B
    genes = ["CXCL8", "IGFBP5", "EGR1", "GADD45A", "SNAPC1"]
    plotGeneDimReduc(genes, "Pf2", pf2Points, dataDF, f, ax[0:5])
    plotGeneDimReduc(genes, "PCA", pcaPoints, dataDF, f, ax[5:10])

    return f
