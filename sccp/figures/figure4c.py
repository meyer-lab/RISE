"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup, 
                     plotCmpUMAP, openPf2, 
                     flattenData, flattenWeightedProjs, 
                     plotCellTypeUMAP, plotCmpPerCellType, 
                     plotGenePerCellType)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import umap 
import os
from os.path import join
import pandas as pd


path_here = os.path.dirname(os.path.dirname(__file__))


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
    pf2Points = umap.UMAP(random_state=1).fit(projs)
    
    plotCellTypeUMAP(pf2Points, dataDF, ax[0])
        
    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    comps = ["Cmp. 5", "Cmp. 12", "Cmp. 20", "Cmp. 30"]
    for i, comp in enumerate(comps):
        plotCmpPerCellType(weightedProjDF, comp, ax[i+1])

    genes = ["NKG7", "GNLY", "MS4A1", "CD79A"]
    for i, gene in enumerate(genes):
        plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Budesonide", "CTRL3"]))], gene, ax[(2*i)+5])
        plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Triamcinolone Acetonide", "CTRL1"]))], gene, ax[(2*i)+6])
    
    
    return f