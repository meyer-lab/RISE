import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ...factorization import pf2_pca_r2x


def plot_r2x(data, rank_vec, ax: Axes):
    """Creates R2X plot for parafac2 tensor decomposition and pca"""
    r2xError = pf2_pca_r2x(data, rank_vec)
    labelNames = ["Fit: Pf2", "Fit: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["o", "o"]
    for i in range(2):
        ax.scatter(
            rank_vec,
            r2xError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )
    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank_vec[-1], num=6, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )


def plot_avegene_per_celltype(adata, genes, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for all conditions"""
    genesV = adata[:, genes]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    df = data.groupby(["Condition", "Cell Type", "Gene"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"})
    sns.boxplot(
        data=df,
        x="Gene",
        y="Average Gene Expression",
        hue="Cell Type",
        ax=ax,
        fliersize=0,
    )


def plot_avegene_per_category(
    conds, categoryCond, gene, adata, ax, mean=True, cellType="Cell Type", swarm=False
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = adata[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=gene).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    if mean is True:
        df = df.groupby(["Condition", "Cell Type", "Gene"], observed=False).mean()

    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"}).reset_index()

    df["Condition"] = np.where(df["Condition"].isin(conds), df["Condition"], "Other")
    for i in conds:
        df = df.replace({"Condition": {i: categoryCond}})

    if swarm is False:
        sns.boxplot(
            data=df.loc[df["Gene"] == gene],
            x="Cell Type",
            y="Average Gene Expression For Drugs",
            hue="Condition",
            ax=ax,
            showfliers=False,
        )
    else:
        sns.stripplot(
            data=df.loc[df["Gene"] == gene],
            x="Cell Type",
            y="Average Gene Expression For Drugs",
            hue="Condition",
            ax=ax,
        )

    ax.set(title=gene)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)


def avegene_per_status(X: anndata.AnnData, gene: str, cellType="Cell Type"):
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

    return df


def gene_plot_cells(
    X: anndata.AnnData,
    hue: str,
    ax: Axes,
    unique=None,
    average=False,
    kde=False,
    cellType="Cell Type",
):
    """Plots two genes on either a per cell or per cell type basis"""
    assert X.shape[1] == 2
    genes = X.var_names
    dataDF = X.to_df()
    dataDF = dataDF.subtract(X.var["means"].values)
    dataDF[hue] = X.obs[hue].values
    dataDF["Cell Type"] = X.obs[cellType].values
    alpha = 1

    if average:
        dataDF = dataDF.groupby([hue], observed=True).mean().reset_index()
        alpha = 1

    if unique is not None:
        dataDF[hue] = dataDF[hue].astype(str)
        dataDF.loc[~dataDF[hue].isin(unique), hue] = "Other"

    sns.scatterplot(data=dataDF, x=genes[0], y=genes[1], hue=hue, ax=ax, alpha=alpha)
    if kde:
        sns.kdeplot(
            data=dataDF,
            x=genes[0],
            y=genes[1],
            hue=hue,
            levels=5,
            fill=True,
            alpha=0.3,
            cut=2,
            ax=ax,
        )


def plot_cell_gene_corr(
    X: anndata.AnnData,
    hue: str,
    cells: list,
    ax: Axes,
    unique=None,
    cellType="Cell Type",
):
    """Plots two genes on either a per cell or per cell type basis"""
    assert X.shape[1] == 2
    genes = X.var_names
    dataDF = X.to_df()
    dataDF = dataDF.subtract(X.var["means"].values)
    dataDF[hue] = X.obs[hue].values
    dataDF["Cell Type"] = X.obs[cellType].values
    alpha = 0.3

    dataDF = dataDF.groupby([hue, "Cell Type"], observed=True).mean().reset_index()
    alpha = 1

    corrDF = pd.DataFrame()
    for cond in dataDF[hue].unique():
        cell_gene1 = dataDF.loc[
            (dataDF[hue] == cond) & (dataDF["Cell Type"] == cells[0])
        ][genes[0]].values
        cell_gene2 = dataDF.loc[
            (dataDF[hue] == cond) & (dataDF["Cell Type"] == cells[1])
        ][genes[1]].values
        corrDF = pd.concat(
            [
                corrDF,
                pd.DataFrame(
                    {
                        hue: cond,
                        cells[0] + " " + genes[0]: cell_gene1,
                        cells[1] + " " + genes[1]: cell_gene2,
                    }
                ),
            ]
        )

    if unique is not None:
        corrDF[hue] = corrDF[hue].astype(str)
        corrDF.loc[~corrDF[hue].isin(unique), hue] = "Other"

    sns.scatterplot(
        data=corrDF,
        x=cells[0] + " " + genes[0],
        y=cells[1] + " " + genes[1],
        hue=hue,
        ax=ax,
        alpha=alpha,
    )


def cell_count_perc_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment"""

    grouping = [celltype, "Condition"]
    
    df = X.obs[grouping].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby(grouping, observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    dfCellType["Cell Type Percentage"] = 0.0
    for cond in np.unique(df["Condition"]):
        dfCellType.loc[dfCellType["Condition"] == cond, "Cell Type Percentage"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Count"].to_numpy()
        )
    
    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)

    return dfCellType


def cell_count_perc_lupus_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment"""
    grouping_all = [celltype, "Condition", "SLE_status", "Processing_Cohort", "condition_unique_idxs"]
    grouping = [celltype, "Condition"]

    df = X.obs[grouping_all].reset_index(drop=True)
    status_mapping = X.obs.groupby("Condition", observed=False)["SLE_status"].first()
    cohort_mapping = X.obs.groupby("Condition", observed=False)["Processing_Cohort"].first()
    idx_mapping = X.obs.groupby("Condition", observed=False)["condition_unique_idxs"].first()

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby(grouping, observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    dfCellType["Cell Type Percentage"] = 0.0
    for cond in np.unique(df["Condition"]):
        dfCellType.loc[dfCellType["Condition"] == cond, "Cell Type Percentage"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Count"].to_numpy()
        )
    
    dfCellType["SLE_status"] = dfCellType["Condition"].map(status_mapping)
    dfCellType["Processing_Cohort"] = dfCellType["Condition"].map(cohort_mapping)
    dfCellType["condition_unique_idxs"] = dfCellType["Condition"].map(idx_mapping)
    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)

    return dfCellType

def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)
