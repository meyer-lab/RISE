"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAP, openPf2, flattenData, flattenWeightedProjs, plotCellTypeUMAP
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import umap 
import numpy as np
import os
from os.path import join
import umap.plot 
import seaborn as sns
import pandas as pd

path_here = os.path.dirname(os.path.dirname(__file__))



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    # ax, f = getSetup((16, 14), (3, 4))
    ax, f = getSetup((16, 16), (5, 5))

    # Add subplot labels
    subplotLabel(ax)
    
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    
    
    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)
    
    # genes = ["NKG7", "CCL5", "GNLY", "KLRD1", "GZMA", "GZMH", "GZMK"]
    
    # genes = ["MS4A1", "CD79A"]
    
    # genes = ["CCR7", "HLA-DQA1"]
    
    genes = ["IL7R", "CD14"]
    
    iter=0
    for i, drug in enumerate(data.condition_labels):
        if i < 12:
        # if i >= 12 and i < 24:
        # if i >= 24 and i < 36:
        # if i >= 36:
            df = dataDF.loc[dataDF["Condition"] == drug]
            for j, genez in enumerate(genes):
                sns.boxplot(data=df[[genez, "Cell Type"]], x=genez, y="Cell Type", ax=ax[iter])
                ax[iter].set(title=drug)
                iter+=1


    
    
    

    # _, factors, projs = openPf2(rank, "Thomson")
    # pf2Points = umap.UMAP(random_state=1).fit(projs)
    
    # plotCellTypeUMAP(pf2Points, dataDF, ax[0])
        
    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    
    # sns.boxplot(data=weightedProjDF, x="Cmp. 5", y="Cell Type", ax=ax[1])
    # sns.boxplot(data=weightedProjDF, x="Cmp. 30", y="Cell Type", ax=ax[2])
    
    # df1 = dataDF[(dataDF["Condition"] == "Triamcinolone Acetonide")]
    # df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    
    # df = pd.concat([df1, df2])

    # genes = ["NKG7", "CCL5", "GNLY", "KLRD1", "GZMA", "GZMH", "GZMK"]
    
    # for i, genez in enumerate(genes):
    #     sns.boxplot(data=df[[genez, "Cell Type", "Condition"]], x=genez, y="Cell Type", hue="Condition", ax=ax[i+3])

    
    return f