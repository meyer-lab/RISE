"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs
)
from .commonFuncs.plotGeneral import(
    plotGenePerCellType
)
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    _, factors, projs = openPf2(rank, "Thomson")
    pf2Points = openUMAP(rank, "Thomson", opt=False)

    plotCellTypeUMAP(pf2Points, dataDF, ax[0])

    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    comps = [5, 12, 20, 30]
    for i, comp in enumerate(comps):
        plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
        plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[(2 * i) + 2])

    set1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    set2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]
    set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

    genes = [set1, set2, set3, set4]
    for i in range(len(genes)):
        plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    return f
