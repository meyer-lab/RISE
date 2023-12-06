"""
Thomson: XX
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP, plotGeneUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategCond,
    plotGeneFactors,
    gene_plot_cells,
    plot_cell_gene_corr
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])

    plotCmpUMAP(X, 16, ax[2], 0.2) #pDC
    plotCmpUMAP(X, 19, ax[3], 0.2) #Alpro 
    plotCmpUMAP(X, 20, ax[4], 0.2) #Gluco

    plotGeneFactors(16, X, ax[5], geneAmount=10, top=True)
    plotGeneFactors(19, X, ax[6], geneAmount=10, top=True)
    plotGeneFactors(20, X, ax[7], geneAmount=10, top=True)
    plotGeneFactors(20, X, ax[8], geneAmount=15, top=False)

    geneSet1 = ["FXYD2", "SERPINF1", "RARRES2"]

    plotGenePerCellType(geneSet1, X, ax[9], cellType="Cell Type2")

    X_genes = X[:, ["THBS1", "VEGFA"]].to_memory()
    gene_plot_cells(X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[10], kde=False)

    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    geneSet3 = ["CD163", "NDRG2"]
    for i in range(len(geneSet3)):
        plotGenePerCategCond(glucs, "Gluco", geneSet3[i], X, ax[i+11], cellType="Cell Type2")


    X_genes = X[:, ["CD163", "NDRG2"]].to_memory()
    plot_cell_gene_corr(X_genes, unique=glucs, hue="Condition", cells=["Intermediate Monocytes", "Myeloid DCs"], cellType="Cell Type2", ax=ax[13])


    return f
