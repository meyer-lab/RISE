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
from .commonFuncs.plotPaCMAP import plot_labels_pacmap, plot_wp_pacmap
from .commonFuncs.plotGeneral import (
    plot_avegene_per_celltype,
    plot_avegene_per_category,
    gene_plot_cells,
    plot_cell_gene_corr,
    heatmapGeneFactors,
    cell_count_perc_df
)
from ..stats import wls_stats_comparison


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (6, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")
    cellDF = cell_count_perc_df(X, "Cell Type2")

    plot_labels_pacmap(X, "Cell Type", ax[0])
    plot_labels_pacmap(X, "Cell Type2", ax[1])
    heatmapGeneFactors([15, 19, 20], X, ax[2], geneAmount=5)

    plot_wp_pacmap(X, 15, ax[3], 0.2)  # pDC
    plot_avegene_per_celltype(X, ["FXYD2"], ax[4], cellType="Cell Type2")
    plot_avegene_per_celltype(X, ["SERPINF1"], ax[5], cellType="Cell Type2")
    plot_avegene_per_celltype(X, ["RARRES2"], ax[6], cellType="Cell Type2")

    plot_wp_pacmap(X, 19, ax[7], 0.2)  # Alpro
    X_genes = X[:, ["THBS1", "EREG"]].to_memory()
    X_genes = X_genes[X_genes.obs["Cell Type"] == "DCs", :]
    gene_plot_cells(
        X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[8], kde=False
    )
    ax[8].set(title="Gene Expression in DCs")

    plot_wp_pacmap(X, 20, ax[9], 0.2)  # Gluco
    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    plot_avegene_per_category(glucs, "Gluco", "CD163", X, ax[10], cellType="Cell Type2")
    plot_avegene_per_category(glucs, "Gluco", "MS4A6A", X, ax[11], cellType="Cell Type2")

    X_genes = X[:, ["CD163", "MS4A6A"]].to_memory()
    plot_cell_gene_corr(
        X_genes,
        unique=glucs,
        hue="Condition",
        cells=["Intermediate Monocytes", "Myeloid Suppressors"],
        cellType="Cell Type2",
        ax=ax[12],
    )

    X.obs["Condition_gluc"] = X.obs["Condition"].cat.add_categories("Other")
    X.obs["Condition_gluc"] = X.obs["Condition_gluc"].cat.add_categories(
        "Glucocorticoids"
    )
    X.obs.loc[~X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Other"
    X.obs.loc[X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Glucocorticoids"
    X.obs["Condition_gluc"] = X.obs["Condition_gluc"].cat.remove_unused_categories()

    color_key = np.flip(sns.color_palette(n_colors=2).as_hex())
    plot_labels_pacmap(X, "Condition_gluc", ax[13], color_key=color_key)

    plot_cell_perc_comp_corr(X, cellDF, "Classical Monocytes", 20, ax[14], unique=glucs)
    plot_cell_perc_comp_corr(X, cellDF, "Myeloid Suppressors", 20, ax[15], unique=glucs)

    cell_perc_box(cellDF, glucs, "Glucocorticoids", ax[16])
    plot_wp_pacmap(X, 9, ax[17], 0.2)  # Gluco
    
    set_xy_limits(ax)

    return f


def plot_cell_perc_corr(cellDF, pop1, pop2, ax):
    """Plots correlation of cell percentages against each other"""
    newDF = pd.DataFrame()
    newDF2 = pd.DataFrame()
    newDF[[pop1, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop1][
        ["Cell Type Percentage", "Condition"]
    ]
    newDF2[[pop2, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop2][
        ["Cell Type Percentage", "Condition"]
    ]
    newDF = newDF.merge(newDF2, on="Condition")
    sns.scatterplot(newDF, x=pop1, y=pop2, hue="Condition", ax=ax)


def plot_cell_perc_comp_corr(X, cellDF, pop, comp, ax, unique=None):
    """Plots correlation of cell percentages against each conditions component value"""
    newDF = pd.DataFrame()
    newDF[[pop, "Condition"]] = cellDF.loc[cellDF["Cell Type"] == pop][
        ["Cell Type Percentage", "Condition"]
    ]
    newDF2 = pd.DataFrame(
        {
            "Comp. " + str(comp): X.uns["Pf2_A"][:, comp - 1],
            "Condition": np.unique(X.obs["Condition"]),
        }
    )
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
    sns.boxplot(
        data=cellDF,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Category",
        showfliers=False,
        hue_order=hue_order,
        ax=ax,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
    
    pValDF = wls_stats_comparison(cellDF, 
                                  column_comparison_name="Cell Type Percentage", 
                                  category_name="Category", 
                                  status_name="Other")
    print(pValDF)


def set_xy_limits(ax):
    ax[4].set(ylim=(-0.1, 1.2))
    ax[5].set(ylim=(-0.1, 1.2))
    ax[6].set(ylim=(-0.1, 1.2))
    ax[8].set(xlim=(-0.05, 0.6), ylim=(-0.05, 0.6))
    ax[10].set(ylim=(-0.05, 0.2))
    ax[11].set(ylim=(-0.05, 0.2))
    ax[12].set(xlim=(-0.05, 0.2), ylim=(-0.05, 0.2))
    ax[14].set(xlim=(0, 0.5), ylim=(0, 70))
    ax[15].set(xlim=(0, 0.5), ylim=(0, 70))
    ax[16].set(ylim=(-10, 70))

    