"""
Thomson: XX
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategCond,
    plotGeneFactors,
    gene_plot_cells,
)
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (4, 4))
    #ax, f = getSetup((16, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    # plotfms(X, 30, ax[0])
    # plotR2X_pf2(X, 15, ax[1])

    gateThomsonCells(X)

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])

    # plotLabelsUMAP(X, "doublet", ax[2], cmap='viridis')
    # plotLabelsUMAP(X, "doublet_score", ax[3], cmap='viridis')
    
    plotCmpUMAP(X, 4, ax[2], 0.2)  # NK
    plotCmpUMAP(X, 23, ax[4], 0.2)  # Gluco
    plotCmpUMAP(X, 25, ax[5], 0.2)  # B Cell
 
    plotCmpUMAP(X, 24, ax[5], 0.2)  # Dex Hcl

    plotGeneFactors(3, X, ax[6], geneAmount=10, top=True)
    plotGeneFactors(23, X, ax[7], geneAmount=10, top=False)
    plotGeneFactors(13, X, ax[8], geneAmount=10, top=False)
    #plotGeneFactors(26, X, ax[9], geneAmount=10, top=False)
    #plotGeneFactors(26, X, ax[10], geneAmount=10, top=True)

    geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CD3D"]
    geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    genes = [geneSet1, geneSet2]
    for i in range(len(genes)):
        plotGenePerCellType(genes[i], X, ax[i + 11])

    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    geneSet3 = ["CD163"]
    plotGenePerCategCond(glucs, "Gluco", geneSet3, X, [ax[13]])

    return f
