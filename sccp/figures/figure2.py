"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotBatchUMAP,
    plotCellUMAP,
    plotCmpUMAP,
)
from ..imports.scRNA import import_pancreas
from ..parafac2 import parafac2_nd
import umap
import scanpy as sc
import pandas as pd
from sklearn.decomposition import PCA
import scib


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    pancreas = import_pancreas(tensor=False)
    cellTypes = pancreas.obs["celltype"].values
    sc.pp.pca(pancreas)
    sc.pp.neighbors(pancreas)
    sc.tl.umap(pancreas)
    sc.pl.umap(pancreas, color=['batch'], palette=sc.pl.palettes.vega_20_scanpy, ax=ax[0])

    pancreas = import_pancreas(tensor=False, method="_bbknn")
    sc.pl.umap(pancreas, color=['batch'], palette=sc.pl.palettes.vega_20_scanpy, ax=ax[1])
    #sc.pl.umap(pancreas, color=['celltype'], palette=sc.pl.palettes.vega_20_scanpy, ax=ax[1])

    pancreas = import_pancreas(tensor=True)
    rank = 50
    _, factors, projs, _ = parafac2_nd(pancreas, rank=rank, random_state=1, verbose=True)
    dataDF, _, _ = flattenData(pancreas, factors, projs)

    # UMAP dimension reduction
    umapReduc = umap.UMAP(random_state=1)
    pf2Points = umapReduc.fit_transform(np.concatenate(projs, axis=0))

    # NK, CD4, B, CD8
    umap_DF = pd.DataFrame({"UMAP 1": pf2Points[:, 0], "UMAP 2": pf2Points[:, 1], "Batch": dataDF.Drug, "Cell Type": cellTypes})
    plotBatchUMAP(umap_DF, ax[2])
    plotCellUMAP(umap_DF, ax[3])

    return f
