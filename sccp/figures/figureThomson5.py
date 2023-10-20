"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotGeneUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, dataName="Thomson")

    cd4 = ["IL7R"]
    cd8 = ["CD8A", "CD8B"]
    nk = ["GNLY", "NKG7"]
    mono1 = ["CD14", "LYZ", "MS4A7"]
    mono2 = ["FCGR3A", "CST3"]
    dc = ["CCR7", "HLA-DQA1", "GPR183"]
    b = ["MS4A1", "CD79A"]
    genes = np.concatenate((cd4, cd8, nk, mono1, mono2, dc, b))

    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "Pf2", X, ax[i])

    return f
