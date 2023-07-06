"""Factors from scib paper"""
import numpy as np
from .common import (subplotLabel, getSetup, 
    plotFactors, flattenData, 
    plotLabelAllUMAP, plotCellCount)
from ..imports.scib import import_scib_data
from ..parafac2 import parafac2_nd
import umap

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    data, celltypes = import_scib_data(dataname="ImmuneHuman")
    # data, celltypes = import_scib_data(dataname="ImmuneHumanMouse")
    # data, celltypes = import_scib_data(dataname="Stimulation1")
    # data, celltypes = import_scib_data(dataname="Stimulation2")
    # data, celltypes = import_scib_data(dataname="Pancreas")
    
    rank = 2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )
    
    dataDF = flattenData(data)
    
     # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    plotFactors(factors, data, ax[0:3], trim=(2,))
    
    plotLabelAllUMAP(dataDF["Condition"].values, pf2Points, ax[3])
    plotLabelAllUMAP(celltypes, pf2Points, ax[4])
    
    plotCellCount(dataDF, celltypes, ax[5])

    return f