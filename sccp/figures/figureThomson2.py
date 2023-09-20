"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs,
)
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import seaborn as sns
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 20), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    print(dataDF)



    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    # _, factors, projs = openPf2(rank, "Thomson")
    # pf2Points = openUMAP(rank, "Thomson", opt=False)

    # plotCellTypeUMAP(pf2Points, dataDF, ax[0])

    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[(2 * i) + 2])

    # geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    # genes = [geneSet1, geneSet2]
    # for i in range(len(genes)):
    #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

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
    # geneSet4 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3"]
    # plotGenePerCategCond(["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[0:len(geneSet4)])
    
    geneSet4 = ["CCNB1"]
    # df = dataDF[["CCNB1", "Condition", "Cell Type"]]
    
    # print(np.max(dataDF["CCNB1"]))
    # sns.boxplot(data = df, x="Cell Type", y="CCNB1",showfliers = False, ax=ax[0] )
    
    # geneSet4 = ["CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3", "CPEB1", "CBR3", "MT1H", "MT1G", "PLK1", "FBN1", "CCR7", "TPX2", 
    #             "FAM131C", "MT1M", "PRSS57", "CENPE", "LTA"]





    # geneSet4 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1", "TYMS", "RRM2", "TESC", "CD70", "BANK1", "CD79B",
    # "KIAA0101", "ZBED2", "CTSW", "CXCL9", "CCNA2", "CXCL10", "MKI67"]
    
    
    df = dataDF[["CCNB1", "Cell Type", "Condition"]]

    for i, celltypes in enumerate(np.unique(df["Cell Type"])):
    
        data = df.loc[df["Cell Type"] == celltypes]
        print(np.mean(data.loc[])
        sns.boxplot(data=data, x="Condition", y="CCNB1", showfliers=False, hue="Condition", ax=ax[i])
        # ax[i].set(title)

        
    # sns.boxplot()
    
    # plotGenePerCategCond(
    #     ["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[0:20])





    return f
