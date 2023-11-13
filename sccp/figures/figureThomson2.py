"""
Thomson: XX
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond, plotfms
from ..gating import gateThomsonCells
import scanpy as sc
import numpy as np
from ..parafac2 import pf2_fms

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    X = openPf2(rank, "Thomson")

    # plotfms(X, 30, ax[0])
    gateThomsonCells(X)
    plotLabelsUMAP(X, "Cell Type", ax[1])
    plotLabelsUMAP(X, "Cell Type2", ax[2])
    plotCmpUMAP(X, 3, ax[3], 0.2)  # NK
    plotCmpUMAP(X, 29, ax[4], 0.2)  # Gluco
    plotCmpUMAP(X, 12, ax[5], 0.2)  # B Cell
    plotCmpUMAP(X, 23, ax[6], 0.2)  # Dex Hcl

    # weightedProjDF = flattenWeightedProjs(data, factors[1], projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors[1], pf2Points, projs, ax[(2 * i) + 2])

    geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CD3D"]
    geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    genes = [geneSet1, geneSet2]
    for i in range(len(genes)):
        plotGenePerCellType(genes[i], X, ax[i + 7])

    # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    geneSet3 = ["CD163"]
    plotGenePerCategCond(glucs, "Gluco", geneSet3, X, [ax[9]])

    # geneSet4 = ["VPREB3", "FAM111B"]
    # plotGenePerCategCond(
    #     ["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[13:15]
    # )

    return f
