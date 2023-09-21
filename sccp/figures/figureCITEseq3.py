"""
Hamad CITEseq dataset
"""
import numpy as np
import umap
from parafac2 import parafac2_nd

from .common import (
    subplotLabel,
    getSetup,
    saveUMAP,
    openUMAP
)
from ..imports.citeseq import import_citeseq, import_citeseqProt
from .commonFuncs.plotUMAP import plotGeneUMAP

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    # data = import_citeseq()
    dataDF = import_citeseqProt()
    rank=40
    
    
    # weight, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )

    # pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    # saveUMAP(pf2Points, rank, "CITEseq")
 
    pf2Points = openUMAP(rank, "CITEseq", opt=False)
    
    genes = np.unique(dataDF.drop(columns="Condition").columns)
    
    # genes = genes[0:24]
    # genes = genes[24:48]
    # genes = genes[50:75]
    # genes = genes[75:100]
    genes = genes[100:]
    
    
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:25])

    return f 