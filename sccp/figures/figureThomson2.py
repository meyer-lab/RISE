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
    heatmapGeneFactors,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 18), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])
    heatmapGeneFactors([15, 19, 20], X, ax[2], geneAmount=5)

    plotCmpUMAP(X, 15, ax[3], 0.4)  # pDC
    geneSet1 = ["FXYD2", "SERPINF1", "RARRES2"]
    plotGenePerCellType(geneSet1, X, ax[4], cellType="Cell Type2")

    plotCmpUMAP(X, 19, ax[5], 0.4)  # Alpro
    X_genes = X[:, ["THBS1", "EREG"]].to_memory()
    X_genes = X_genes[X_genes.obs["Cell Type"] == "DCs", :]
    gene_plot_cells(
        X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[6], kde=False
    )
    ax[6].set(title="Gene Expression in DCs")

    plotCmpUMAP(X, 20, ax[7], 0.4)  # Gluco
    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    plotGenePerCategCond(glucs, "Gluco", "CD163", X, ax[8], cellType="Cell Type2")
    plotGenePerCategCond(glucs, "Gluco", "NDRG2", X, ax[9], cellType="Cell Type2")

    X_genes = X[:, ["CD163", "NDRG2"]].to_memory()
    plot_cell_gene_corr(
        X_genes,
        unique=glucs,
        hue="Condition",
        cells=["Intermediate Monocytes", "Myeloid DCs"],
        cellType="Cell Type2",
        ax=ax[10],
    )

    ax[6].set(xlim=(-0.05, 0.6), ylim=(-0.05, 0.6))
    ax[8].set(ylim=(-0.05, 0.2))
    ax[9].set(ylim=(-0.05, 0.3))
    ax[10].set(xlim=(-0.05, 0.3), ylim=(-0.05, 0.3))

    return f
