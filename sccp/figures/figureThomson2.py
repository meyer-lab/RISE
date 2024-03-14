"""
Thomson: XX
"""
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP, plotGeneUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategCond,
    gene_plot_cells,
    plot_cell_gene_corr,
    heatmapGeneFactors,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 22.5), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")
    cellDF = getCellCountDF(X, "Cell Type2")

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])
    heatmapGeneFactors([15, 19, 20], X, ax[2], geneAmount=5)

    plotCmpUMAP(X, 15, ax[3], 0.2)  # pDC
    geneSet1 = ["FXYD2", "SERPINF1", "RARRES2"]
    plotGenePerCellType(geneSet1, X, ax[4], cellType="Cell Type2")

    plotCmpUMAP(X, 19, ax[5], 0.2)  # Alpro
    X_genes = X[:, ["THBS1", "EREG"]].to_memory()
    X_genes = X_genes[X_genes.obs["Cell Type"] == "DCs", :]
    gene_plot_cells(
        X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[6], kde=False
    )
    ax[6].set(title="Gene Expression in DCs")

    plotCmpUMAP(X, 20, ax[7], 0.2)  # Gluco
    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    plotGenePerCategCond(glucs, "Gluco", "CD163", X, ax[8], cellType="Cell Type2")
    plotGenePerCategCond(glucs, "Gluco", "MS4A6A", X, ax[9], cellType="Cell Type2")

    X_genes = X[:, ["CD163", "MS4A6A"]].to_memory()
    plot_cell_gene_corr(
        X_genes,
        unique=glucs,
        hue="Condition",
        cells=["Intermediate Monocytes", "Myeloid Suppressors"],
        cellType="Cell Type2",
        ax=ax[10],
    )


    X.obs['Condition_gluc'] = X.obs['Condition'].cat.add_categories('Other')
    X.obs['Condition_gluc'] = X.obs['Condition_gluc'].cat.add_categories('Glucocorticoids')
    X.obs.loc[~X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Other"
    X.obs.loc[X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Glucocorticoids"
    X.obs['Condition_gluc'] = X.obs['Condition_gluc'].cat.remove_unused_categories()
    
    color_key = np.flip(sns.color_palette(n_colors=2).as_hex())
    plotLabelsUMAP(X, "Condition_gluc", ax[11], color_key=color_key)

    plot_cell_perc_comp_corr(X, cellDF, "Classical Monocytes", 20, ax[12], unique=glucs)
    plot_cell_perc_comp_corr(X, cellDF, "Myeloid Suppressors", 20, ax[13], unique=glucs)

    cell_perc_box(cellDF, glucs, "Glucocorticoids", ax[14])
    plotCmpUMAP(X, 9, ax[15], 0.2)  # Gluco

    ax[6].set(xlim=(-0.05, 0.6), ylim=(-0.05, 0.6))
    ax[8].set(ylim=(-0.05, 0.2))
    ax[9].set(ylim=(-0.05, 0.2))
    ax[10].set(xlim=(-0.05, 0.2), ylim=(-0.05, 0.2))
    ax[12].set(xlim=(0, 0.5), ylim=(0, 70))
    ax[13].set(xlim=(0, 0.5), ylim=(0, 70))
    ax[14].set(ylim=(-10, 70))

    return f



def getCellCountDF(X, celltype="Cell Type", cellPerc=True):
    """Returns DF with percentages of cells"""

    df = X.obs[["Cell Type", "Condition", "Cell Type2"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby([celltype, "Condition"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    dfCellType["Count"] = dfCellType["Count"].astype("float")

    if cellPerc is True:
        for i, cond in enumerate(np.unique(df["Condition"])):
            dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
                100
                * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
                / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
            )
        dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)

    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)

    return dfCellType


def plot_cell_perc_corr(cellDF, pop1, pop2, ax):
    newDF = pd.DataFrame()
    newDF2 = pd.DataFrame()
    newDF[[pop1, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop1][["Cell Type Percentage", "Condition"]]
    newDF2[[pop2, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop2][["Cell Type Percentage", "Condition"]]
    newDF = newDF.merge(newDF2, on="Condition")
    sns.scatterplot(newDF, x=pop1, y=pop2, hue="Condition", ax=ax)


def plot_cell_perc_comp_corr(X, cellDF, pop, comp, ax, unique=None):
    newDF = pd.DataFrame()
    newDF[[pop, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop][["Cell Type Percentage", "Condition"]]
    newDF2 = pd.DataFrame({"Comp. " + str(comp): X.uns["Pf2_A"][:, comp - 1], "Condition": np.unique(X.obs["Condition"])})
    newDF = newDF.merge(newDF2, on="Condition")

    if unique is not None:
        newDF["Condition"] = newDF["Condition"].astype(str)
        newDF.loc[~newDF["Condition"].isin(unique), "Condition"] = "Other"
    
    sns.scatterplot(newDF, x="Comp. " + str(comp), y=pop, hue="Condition", ax=ax)
 

def cell_perc_box(cellDF, unique, uniqueLabel, ax):
    """Plots percentages of cells against each other"""
    cellDF["Category"] = uniqueLabel
    cellDF.loc[~cellDF.Condition.isin(unique), "Category"] = "Other"
    hue_order = ["Other", uniqueLabel]
    sns.boxplot(data=cellDF, x="Cell Type", y="Cell Type Percentage", hue="Category", showfliers=False, hue_order=hue_order, ax=ax)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
