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
    ax, f = getSetup((15, 42), (20, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    comps = [22, 33, 47, 48, 23, 31, 43]
    genes = top_bot_genes(X, cmp=comps[6], geneAmount=10)

    for i, gene in enumerate(genes):
        plotGenePerStatus(X, gene, ax[i], cellType="leiden")

    return f


def top_bot_genes(X, cmp, geneAmount=5):
    """Saves most pos/negatively genes"""
    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=["Component"]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by="Component")
    
    print(df)
    
    top = df.iloc[-geneAmount:, 0].values
    bot = df.iloc[:geneAmount, 0].values
    all_genes = np.concatenate([bot, top])

    return all_genes


def plotGenePerStatus(X, gene, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene].to_memory()
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(dataDF, id_vars=["Cell Type", "Condition"], value_vars=gene).rename(
        columns={"variable": "Gene", "value": "Value"}
    )

    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Average Gene Expression",
        hue="Condition",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)
