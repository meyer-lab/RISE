"""
Lupus: Average gene expression stratified by cell type and status
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .commonFuncs.plotFactors import bot_top_genes
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    genes = bot_top_genes(X, cmp=27, geneAmount=5)

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status_per_cluster(
            X, gene, ax[i], clusterName1="44", cellType="leiden"
        )

    return f


def plot_avegene_per_status_per_cluster(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    clusterName1,
    clusterName2=None,
    cellType="Cell Type",
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    if clusterName2 is None:
        dfClust = df.loc[df["Cell Type"] == clusterName1]
        clust_list = dfClust["Cell Type"].to_numpy()
        dfOther = df.loc[df["Cell Type"] != clusterName1]
        other_list = np.repeat("Other", dfOther.shape[0])

        dfClust = pd.concat([dfClust, dfOther]).reset_index(drop=True)
        dfClust["Cell Type"] = np.concatenate([clust_list, other_list])

    else:
        dfClust = df.loc[
            (df["Cell Type"] == clusterName1) & (df["Cell Type"] == clusterName2)
        ]
        clust_list = dfClust["Cell Type"].to_numpy()
        dfOther = df.loc[
            (df["Cell Type"] != clusterName1) & (df["Cell Type"] != clusterName2)
        ]
        other_list = np.repeat("Other", dfOther.shape[0])

        dfClust = pd.concat([dfClust, dfOther]).reset_index(drop=True)
        dfClust["Cell Type"] = np.concatenate([clust_list, other_list])

    sns.boxplot(
        data=dfClust,
        x="Cell Type",
        y="Average Gene Expression",
        hue="Status",
        ax=ax,
        showfliers=False,
    )

    ax.set(
        title=gene,
        yticks=np.linspace(
            0, np.max(dfClust["Average Gene Expression"]) + 0.00005, num=5
        ),
    )
