"""
CITEseq: Plotting genes per component
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import pandas as pd
import seaborn as sns
from .commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 42), (20, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    comps = [22, 33, 47, 48, 23, 31, 43]
    genes = bot_top_genes(X, cmp=comps[6], geneAmount=10)

    for i, gene in enumerate(genes):
        plotGenePerStatus(X, gene, ax[i], cellType="leiden")

    return f


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
