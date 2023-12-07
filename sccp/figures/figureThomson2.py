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
    plot_cell_gene_corr,
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

    plotCmpUMAP(X, 16, ax[2], 0.2)  # pDC
    plotGeneFactors(15, X, ax[3], geneAmount=10, top=True)
    geneSet1 = ["FXYD2", "SERPINF1", "RARRES2"]
    plotGenePerCellType(geneSet1, X, ax[4], cellType="Cell Type2")


    plotCmpUMAP(X, 19, ax[5], 0.2)  # Alpro
    plotGeneFactors(19, X, ax[6], geneAmount=10, top=True)
    X_genes = X[:, ["THBS1", "VEGFA"]].to_memory()
    X_genes = X_genes[X_genes.obs["Cell Type"] == "DCs", :]
    gene_plot_cells(
        X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[7], kde=False
    )
    ax[7].set(title="Gene Expression in DCs")

    plotCmpUMAP(X, 20, ax[8], 0.2)  # Gluco
    plotGeneFactors(20, X, ax[9], geneAmount=10, top=True)

    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    plotGenePerCategCond(glucs, "Gluco", "CD163", X, ax[10], cellType="Cell Type2")
        
    plotGeneFactors(20, X, ax[11], geneAmount=10, top=False)
    plotGenePerCategCond(glucs, "Gluco", "NDRG2", X, ax[12], cellType="Cell Type2")

    X_genes = X[:, ["CD163", "NDRG2"]].to_memory()
    plot_cell_gene_corr(
        X_genes,
        unique=glucs,
        hue="Condition",
        cells=["Intermediate Monocytes", "Myeloid DCs"],
        cellType="Cell Type2",
        ax=ax[13],
    )
    X_genes = X[:, ["VEGFA", "TNF"]].to_memory()


    return f
