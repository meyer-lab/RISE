import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..factorization import rise_pca_r2x


def plot_r2x(data, rank_vec, ax: Axes):
    """Plot variance explained (R²X) for RISE and PCA across different ranks.

    This visualization helps determine the optimal number of components by showing
    how variance explained increases with rank. The elbow point where the curve
    flattens indicates a good balance between model complexity and explanatory power.

    Parameters
    ----------
    data : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] for RISE decomposition.
    rank_vec : array-like of int
        Array of rank values to test (e.g., [1, 5, 10, 15, 20, 25, 30]).
        Each rank represents a different number of components.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    """
    r2xError = rise_pca_r2x(data, rank_vec)
    labelNames = ["Fit: RISE", "Fit: PCA"]
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
    ax.legend()


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

    meanDF = dataDF.groupby([hue, "Cell Type"], observed=True).mean().reset_index()
    pivoted = meanDF.pivot(index=hue, columns="Cell Type", values=genes)

    col1 = (genes[0], cells[0])
    col2 = (genes[1], cells[1])

    x_name = f"{cells[0]} {genes[0]}"
    y_name = f"{cells[1]} {genes[1]}"

    if col1 in pivoted.columns and col2 in pivoted.columns:
        corrDF = pd.DataFrame(
            {
                hue: pivoted.index,
                x_name: pivoted[col1].values,
                y_name: pivoted[col2].values,
            }
        ).dropna()
    else:
        corrDF = pd.DataFrame(columns=[hue, x_name, y_name])

    if unique is not None:
        corrDF[hue] = corrDF[hue].astype(str)
        corrDF.loc[~corrDF[hue].isin(unique), hue] = "Other"

    sns.scatterplot(
        data=corrDF,
        x=x_name,
        y=y_name,
        hue=hue,
        ax=ax,
        alpha=1.0,
    )


def cell_count_perc_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment"""
    grouping = [celltype, "Condition"]
    df = X.obs[grouping].reset_index(drop=True)

    dfCellType = (
        df.groupby(grouping, observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    dfCellType["Cell Type Percentage"] = (
        100
        * dfCellType["Cell Count"]
        / dfCellType.groupby("Condition")["Cell Count"].transform("sum")
    )

    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)
    return dfCellType


def cell_count_perc_lupus_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment with Lupus metadata"""
    dfCellType = cell_count_perc_df(X, celltype=celltype)

    status_mapping = X.obs.groupby("Condition", observed=False)["SLE_status"].first()
    cohort_mapping = X.obs.groupby("Condition", observed=False)[
        "Processing_Cohort"
    ].first()
    idx_mapping = X.obs.groupby("Condition", observed=False)[
        "condition_unique_idxs"
    ].first()

    dfCellType["SLE_status"] = dfCellType["Condition"].map(status_mapping)
    dfCellType["Processing_Cohort"] = dfCellType["Condition"].map(cohort_mapping)
    dfCellType["condition_unique_idxs"] = dfCellType["Condition"].map(idx_mapping)

    return dfCellType


def rotate_xaxis(ax, rotation=90):
    """Rotates text for x-axis"""
    ax.tick_params(axis="x", rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text for y-axis"""
    ax.tick_params(axis="y", rotation=rotation)
