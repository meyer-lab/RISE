"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotBatchUMAP,
    plotCellUMAP
)
from ..imports.scRNA import import_pancreas
from ..parafac2 import parafac2_nd
import umap
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    pancreas = import_pancreas(tensor=False)
    cellTypes = pancreas.obs["celltype"].values
    pancreas_pf2 = import_pancreas(tensor=True)
    
    rank = 2
    _, factors, projs, _ = parafac2_nd(pancreas_pf2, rank=rank, random_state=1)
    dataDF  = flattenData(pancreas_pf2)

    # UMAP dimension reduction
    umapReduc = umap.UMAP(random_state=1)
    pf2Points = umapReduc.fit_transform(np.concatenate(projs, axis=0))

    # NK, CD4, B, CD8
    umap_DF = pd.DataFrame({"UMAP 1": pf2Points[:, 0], "UMAP 2": pf2Points[:, 1], "Batch": dataDF["Condition"].values, "Cell Type": cellTypes})
    plotBatchUMAP(umap_DF, ax[2])
    plotCellUMAP(umap_DF, ax[3])

    # Import of single cells: [Drug, Cell, Gene]
    cellTypes = pancreas.obs["celltype"].values
    sc.pp.pca(pancreas)
    sc.pp.neighbors(pancreas)
    sc.tl.umap(pancreas)
    umap_DF = UMAP_DFify(pancreas, dataDF["Condition"].values, cellTypes)
    plotBatchUMAP(umap_DF, ax[0])
    plotCellUMAP(umap_DF, ax[1])

    pancreas = import_pancreas(tensor=False, method="_bbknn")
    umap_DF = UMAP_DFify(pancreas, dataDF["Condition"].values, cellTypes)
    plotBatchUMAP(umap_DF, ax[4])
    plotCellUMAP(umap_DF, ax[5])

    pancreas = import_pancreas(tensor=False, method="_harmony")
    umap_DF = UMAP_DFify(pancreas, dataDF["Condition"].values, cellTypes)
    plotBatchUMAP(umap_DF, ax[6])
    plotCellUMAP(umap_DF, ax[7])

    return f


def UMAP_DFify(data, drug, celltypes):
    """Provides dataframe version of UMAP data from anndata source"""
    return pd.DataFrame({"UMAP 1": data.obsm["X_umap"][:, 0], "UMAP 2": data.obsm["X_umap"][:, 1], "Batch": drug, "Cell Type": celltypes})
