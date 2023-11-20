import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata
from matplotlib.axes import Axes

from ...factorization import pf2_r2x
from ...crossVal import CrossVal
from ...factorization import pf2_fms


def plotR2X(data, rank, ax: Axes):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = pf2_r2x(data, rank)

    rank_vec = np.arange(1, rank + 1)
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
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )

    ax.legend()


def plotR2X_pf2(data, rank, ax: Axes):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = pf2_r2x(data, rank)

    rank_vec = np.arange(1, rank + 1)
    ax.scatter(
        rank_vec,
        r2xError,
        c="k",
        s=30.0,
    )

    ax.set(
        ylabel="Factor Match Score",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )


def plotfms(data, rank, ax: Axes):
    """Creates R2X plot for parafac2 tensor decomposition"""
    fms_vec = pf2_fms(data, rank)

    rank_vec = np.arange(1, rank + 1)

    ax.scatter(
        rank_vec,
        fms_vec,
        c="k",
        s=30.0,
    )

    ax.set(
        ylabel="Factor Match Score",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(0, np.max(np.append(fms_vec[0], fms_vec[1])) + 0.01, num=5),
    )


def plotCV(data, rank, trainPerc, ax: Axes):
    """Creates variance explained plot for parafac2 tensor decomposition CV"""
    cvError = CrossVal(data, rank, trainPerc=trainPerc)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["CV: Pf2", "CV: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["o", "o"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            cvError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(0, np.max(np.append(cvError[0], cvError[1])) + 0.01, num=5),
    )

    ax.legend()


def plotCellTypePerExpCount(dataDF, condition, ax: Axes):
    """Plots historgram of cell counts per experiment"""
    sns.histplot(data=dataDF, x="Cell Type", hue="Cell Type", ax=ax)
    ax.set(title=condition)


def plotCellTypePerExpPerc(dataDF, condition, ax: Axes):
    """Plots historgram of cell types percentages per experiment"""
    df = dataDF.groupby(["Cell Type"]).size().reset_index(name="Count")
    perc = df["Count"].values / np.sum(df["Count"].values)
    df["Count"] = perc

    sns.barplot(data=df, x="Cell Type", y="Count", ax=ax)
    ax.set(title=condition)


def plotGenePerCellType(genes, adata, ax):
    """Plots average gene expression across cell types for all conditions"""
    genesV = adata[:, genes]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs["Cell Type"].values
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    df = data.groupby(["Condition", "Cell Type", "Gene"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"})
    sns.stripplot(
        data=df,
        x="Gene",
        y="Average Gene Expression For Drugs",
        hue="Cell Type",
        dodge=True,
        jitter=False,
        ax=ax,
    )


def plotGenePerCategCond(conds, categoryCond, genes, adata, axs, mean=True):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = adata[:, genes]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs["Cell Type"].values

    df = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    if mean is True:
        df = df.groupby(["Condition", "Cell Type", "Gene"], observed=False).mean()

    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"}).reset_index()

    df["Condition"] = np.where(df["Condition"].isin(conds), df["Condition"], "Other")
    for i in conds:
        df = df.replace({"Condition": {i: categoryCond}})

    for i, gene in enumerate(genes):
        sns.boxplot(
            data=df.loc[df["Gene"] == gene],
            x="Cell Type",
            y="Average Gene Expression For Drugs",
            hue="Condition",
            ax=axs[i],
        )
        axs[i].set(title=gene)


def plotGeneFactors(
    cmp: int, dataIn: anndata.AnnData, ax: Axes, geneAmount: int = 20, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=dataIn.varm["Pf2_C"][:, cmp - 1], index=dataIn.var_names, columns=[cmpName]
    )

    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=ax
        )
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)


def population_bar_chart(
    adata: anndata.AnnData, cellType: str, category: str, ax: Axes
):
    """Plots proportion of cells by type stratified by an identifying condition or patient attribute (i.e. Lupus Status)"""
    cellDF = pd.crosstab(adata.obs[category], adata.obs[cellType], normalize="index")
    cellDF.plot.bar(ax=ax, stacked=True).legend(loc="upper right")
    ax.set(ylim=(0, 1), ylabel="Proportion of Cells")


def cell_comp_hist(X, category: str, comp: int, unique, ax: Axes):
    """Plots weighted projections of each cell according to category"""
    w_proj = X.obsm["weighted_projections"][:, comp - 1]
    obsDF = pd.DataFrame({category: X.obs[category].astype(str).values})
    w_proj = X.obsm["weighted_projections"][:, comp - 1]
    if category is not None:
        if unique is not None:
            obsDF.loc[obsDF[category] != unique, category] = "Other"
        labels = obsDF[category]
        # obsDF[category] = obsDF[category].astype(str)
        histDF = pd.DataFrame({"Component " + str(comp): w_proj, category: labels})
        sns.histplot(
            data=histDF,
            x="Component " + str(comp),
            hue=category,
            kde=True,
            ax=ax,
            stat="density",
            common_norm=False,
        )


def gene_plot_cells(
    X: anndata.AnnData, hue: str, ax: Axes, unique=None, average=False, kde=False
):
    """Plots two genes on either a per cell or per cell type basis"""
    assert X.shape[1] == 2
    genes = X.var_names
    sc.pp.subsample(X, fraction=1, random_state=0)
    dataDF = X.to_df()
    dataDF[hue] = X.obs[hue].values
    alpha = 0.3

    if average:
        dataDF = dataDF.groupby([hue], observed=True).mean().reset_index()
        alpha = 1

    if unique is not None:
        dataDF[hue] = dataDF[hue].astype(str)
        dataDF.loc[dataDF[hue] != unique, hue] = "Other"

    sns.scatterplot(
        data=dataDF, x=genes[0], y=genes[1], hue=hue, ax=ax, size=-0.1, alpha=alpha
    )
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


def gene_plot_conditions(X, condition: str, genes, ax: Axes, hue=None, unique=None):
    """Plots two genes on either a per cell or per cell type basis"""
    adata = X[:, [genes[0], genes[1]]]
    sc.pp.subsample(adata, fraction=0.01, random_state=0)

    dataDF = pd.DataFrame(columns=genes, data=adata.X)
    dataDF[condition] = adata.obs[condition].values
    dataDF[condition] = dataDF[condition].astype("str")
    if hue:
        dataDF[hue] = adata.obs[hue].values
        dataDF[hue] = dataDF[hue].astype("str")
        dataDF = dataDF.groupby([condition, hue]).mean()
    else:
        dataDF = dataDF.groupby([condition]).mean()
    if unique is not None:
        dataDF[condition] = dataDF[condition].astype(str)
        dataDF.loc[dataDF[condition] != unique, condition] = "Other"
    if hue is not None:
        sns.scatterplot(
            data=dataDF, x=genes[0], y=genes[1], hue=hue, ax=ax, size=-0.1, alpha=5
        )
    else:
        sns.scatterplot(
            data=dataDF, x=genes[0], y=genes[1], ax=ax, size=-0.1, alpha=0.2
        )


def geneSig_plot_cells(
    X, comps: list[int], hue: str, ax: Axes, unique=None, average=False, kde=False
):
    """Plots two genes on either a per cell or per cell type basis"""

    geneSigDF = pd.DataFrame()
    geneVecs = X.varm["Pf2_C"][:, comps]
    for i, _ in enumerate(comps):
        geneSigDF[str(comps[i])] = np.matmul(X.X, geneVecs[:, i])

    geneSigDF[hue] = X.obs[hue].values
    geneSigDF = geneSigDF.sample(n=10000)
    alpha = 0.3

    if unique is not None:
        geneSigDF[hue] = geneSigDF[hue].astype(str)
        geneSigDF.loc[geneSigDF[hue] != unique, hue] = "Other"
    if average:
        geneSigDF = geneSigDF.groupby([hue], observed=True).mean()
        alpha = 1
    sns.scatterplot(
        data=geneSigDF,
        x=str(comps[0]),
        y=str(comps[1]),
        hue=hue,
        ax=ax,
        size=-0.1,
        alpha=alpha,
    )
    if kde:
        sns.kdeplot(
            data=geneSigDF,
            x=str(comps[0]),
            y=str(comps[1]),
            hue=hue,
            levels=5,
            fill=True,
            alpha=0.3,
            cut=2,
            ax=ax,
        )
    ax.set(
        xlabel="Comp. " + str(comps[0]) + " Signature",
        ylabel="Comp. " + str(comps[1]) + " Signature",
    )
