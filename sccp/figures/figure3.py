"""
Figure 3
"""

import numpy as np
import pandas as pd
import seaborn as sns
import anndata
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
    cell_count_perc_df,
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
)
from .figure2f_h import groupDrugs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((30, 30), (5, 5))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    cellDF = cell_count_perc_df(X, "Cell Type2")
    plot_labels_pacmap(X, "Cell Type2", ax[0])

    drugNames = groupDrugs(X, "Condition")
    plot_condition_factors(X, ax[1], drugNames, ThomsonNorm=True, groupConditions=True)
    plot_gene_factors(X, ax[2])
    ax[2].yaxis.set_ticklabels([])

    plot_wp_pacmap(X, 15, ax[3], 0.2)
    plot_avegene_per_celltype(X, ["FXYD2"], ax[4], cellType="Cell Type2")
    plot_avegene_per_celltype(X, ["SERPINF1"], ax[5], cellType="Cell Type2")
    plot_avegene_per_celltype(X, ["RARRES2"], ax[6], cellType="Cell Type2")

    plot_wp_pacmap(X, 19, ax[7], 0.2)
    plot_avegene_per_category(
        ["Alprostadil"], "Alpro", "EREG", X, ax[8], cellType="Cell Type", swarm=True
    )
    X_genes = X[:, ["THBS1", "EREG"]].to_memory()
    X_genes = X_genes[X_genes.obs["Cell Type"] == "DCs", :]
    gene_plot_cells(
        X_genes, unique=["Alprostadil"], hue="Condition", ax=ax[9], kde=False
    )
    ax[9].set(title="Gene Expression in DCs")

    plot_wp_pacmap(X, 20, ax[9], 0.2)

    X.obs["Condition_gluc"] = X.obs["Condition"].cat.add_categories("Other")
    X.obs["Condition_gluc"] = X.obs["Condition_gluc"].cat.add_categories(
        "Glucocorticoids"
    )
    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    X.obs.loc[~X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Other"
    X.obs.loc[X.obs["Condition_gluc"].isin(glucs), "Condition_gluc"] = "Glucocorticoids"
    X.obs["Condition_gluc"] = X.obs["Condition_gluc"].cat.remove_unused_categories()
    color_key = np.flip(sns.color_palette(n_colors=2).as_hex())
    plot_labels_pacmap(X, "Condition_gluc", ax[10], color_key=color_key)

    cell_perc_box(cellDF, glucs, "Glucocorticoids", ax[11])

    plot_cell_perc_comp_corr(X, cellDF, "Myeloid Suppressors", 20, ax[12], unique=glucs)

    plot_wp_pacmap(X, 9, ax[13], 0.2)

    plot_avegene_per_category(
        glucs, "Gluco", "MS4A6A", X, ax[14], cellType="Cell Type2"
    )

    X_genes = X[:, ["CD163", "MS4A6A"]].to_memory()
    plot_cell_gene_corr(
        X_genes,
        unique=glucs,
        hue="Condition",
        cells=["Intermediate Monocytes", "Myeloid Suppressors"],
        cellType="Cell Type2",
        ax=ax[15],
    )

    # set_xy_limits(ax)

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


def set_xy_limits(ax):
    ax[1].set(ylim=(-0.1, 1.2))
    ax[2].set(ylim=(-0.1, 1.2))
    ax[3].set(ylim=(-0.1, 1.2))

    ax[10].set(ylim=(-0.05, 0.2))
    ax[13].set(xlim=(0, 0.5), ylim=(0, 70))
    ax[14].set(ylim=(-10, 70))
