"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup, 
                     plotCmpUMAP, openPf2, openUMAP,
                     flattenData, flattenWeightedProjs, 
                     plotCellTypeUMAP, plotCmpPerCellType, 
                     plotGenePerCellType)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells

import pandas as pd


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
        plotCmpPerCellType(weightedProjDF, comps[i], ax[i+1])

    genes = ["NKG7", "GNLY", "MS4A1", "CD79A"]
    for i, gene in enumerate(genes):
        plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Budesonide", "CTRL3"]))], gene, ax[(2*i)+5])
        plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Triamcinolone Acetonide", "CTRL1"]))], gene, ax[(2*i)+6])
    
    
    return f