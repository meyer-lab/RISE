"""
Figure 5f_j
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    plot_wp_pacmap(X, 28, ax[2], 0.25)

    genes = ["RETN", "S100A9", "S100A12", "S100A8"]

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status_per_cluster(
            X, gene, ax[i + 2], clusterName1="44", cellType="leiden"
        )
    genes = ["IFITM3"]
    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status_per_cluster(
            X, gene, ax[i + 2], clusterName1="21", clusterName2="30", cellType="leiden"
        )

    plot_pair_gene_factors(X, 22, 28, ax[5])

    return f


def plot_avegene_per_status_per_cluster(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    clusterName1: str,
    clusterName2=None,
    cellType: str = "Cell Type",
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


def plot_pair_gene_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.varm["Pf2_C"][:, cmp1 - 1]], [X.varm["Pf2_C"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
