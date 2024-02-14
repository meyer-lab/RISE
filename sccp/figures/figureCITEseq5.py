"""
CITEseq: Plotting genes per component
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import pandas as pd
import numpy as np
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")
    XX = read_h5ad("/opt/pf2/CITE_Neighbors.h5ad", backed="r")
    X.obs["leiden"] = XX.obs["leiden"]

    comps = [22, 33, 47, 48, 23, 31, 43]
    genes = top_bot_genes(X, cmp=comps[2], geneAmount=1)

    for i, gene in enumerate(genes):
        plotGenePerStatus(X, gene, ax[i], cellType="leiden")
        # ax[i].legend([],[], frameon=False)

    return f


def top_bot_genes(X, cmp, geneAmount=5):
    """Saves most pos/negatively genes"""
    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=["Component"]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by="Component")

    top = df.iloc[-geneAmount:, 0].values
    bot = df.iloc[:geneAmount, 0].values
    all_genes = np.concatenate([top, bot])

    return all_genes


def plotGenePerStatus(X, gene, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    adata = X.to_memory()
    genesV = adata[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(dataDF, id_vars=["Cell Type", "Condition"], value_vars=gene).rename(
        columns={"variable": "Gene", "value": "Value"}
    )

    # df = df.groupby(["Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df,
        x="Condition",
        y="Average Gene Expression",
        hue="Cell Type",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)
