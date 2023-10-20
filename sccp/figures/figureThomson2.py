"""
Thomson: XX
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP, points
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, "Thomson")

    X.obs["Cell Type"] = gateThomsonCells(X)

    points(X.obsm["embedding"], labels=X.obs["Cell Type"].values, ax=ax[0])
    ax[0].set(ylabel="UMAP2", xlabel="UMAP1")

    # weightedProjDF = flattenWeightedProjs(data, factors[1], projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors[1], pf2Points, projs, ax[(2 * i) + 2])

    # geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    # genes = [geneSet1, geneSet2]
    # for i in range(len(genes)):
    #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

    # glucs = [
    #     "Betamethasone Valerate",
    #     "Loteprednol etabonate",
    #     "Budesonide",
    #     "Triamcinolone Acetonide",
    #     "Meprednisone",
    # ]
    # geneSet3 = ["CD163", "ADORA3"]
    # plotGenePerCategCond(glucs, "Gluco", geneSet3, dataDF, ax[11:13])

    # geneSet4 = ["VPREB3", "FAM111B"]
    # plotGenePerCategCond(
    #     ["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[13:15]
    # )

    return f
