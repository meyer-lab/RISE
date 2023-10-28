import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata
from copy import deepcopy
from ...crossVal import CrossVal
from ...decomposition import R2X


def plotR2X(data, rank, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = R2X(data, rank)

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


def plotCV(data, rank, trainPerc, ax):
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


def plotCellTypePerExpCount(dataDF, condition, ax):
    """Plots historgram of cell counts per experiment"""
    sns.histplot(data=dataDF, x="Cell Type", hue="Cell Type", ax=ax)
    ax.set(title=condition)


def plotCellTypePerExpPerc(dataDF, condition, ax):
    """Plots historgram of cell types percentages per experiment"""
    df = dataDF.groupby(["Cell Type"]).size().reset_index(name="Count")
    perc = df["Count"].values / np.sum(df["Count"].values)
    df["Count"] = perc

    sns.barplot(data=df, x="Cell Type", y="Count", ax=ax)
    ax.set(title=condition)


def plotGenePerCellType(genes, dataDF: pd.DataFrame, ax):
    """Plots average gene expression across cell types for all conditions"""
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    df = data.groupby(["Condition", "Cell Type", "Gene"]).mean()
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


def plotGenePerCategCond(conds, categoryCond, genes, dataDF, axs, mean=True):
    """Plots average gene expression across cell types for a category of drugs"""

    df = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    if mean is True:
        df = df.groupby(["Condition", "Cell Type", "Gene"]).mean()

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


def plotGeneFactors(cmp: int, dataIn: anndata.AnnData, axs, geneAmount: int = 20):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=dataIn.varm["Pf2_C"][:, cmp - 1], index=dataIn.var_names, columns=[cmpName]
    )

    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=axs[0])
    sns.barplot(
        data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=axs[1]
    )
    axs[0].tick_params(axis="x", rotation=90)
    axs[1].tick_params(axis="x", rotation=90)


def population_bar_chart(adata: anndata.AnnData, cellType: str, category: str, ax):
    """Plots proportion of cells by type stratified by an identifying condition or patient attribute (i.e. Lupus Status)"""
    cellDF = pd.crosstab(adata.obs[category], adata.obs[cellType], normalize="index")
    cellDF.plot.bar(ax=ax, stacked=True).legend(loc="upper right")
    ax.set(ylim=(0, 1), ylabel="Proportion of Cells")


def cell_comp_hist(X, category: str, comp: int, unique, ax):
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
            data=histDF, x="Component " + str(comp), hue=category, kde=True, ax=ax
        )


def gene_plot_cells(X, genes, hue: str, ax, unique=None, average=False, kde=False):
    """Plots two genes on either a per cell or per cell type basis"""
    adata = X[:, [genes[0], genes[1]]]
    sc.pp.subsample(adata, fraction=0.01, random_state=0)
    dataDF = pd.DataFrame(columns=genes, data=adata.X)
    dataDF[hue] = adata.obs[hue].values
    alpha = 0.3
    if unique is not None:
        dataDF[hue] = dataDF[hue].astype(str)
        dataDF.loc[dataDF[hue] != unique, hue] = "Other"
    if average:
        dataDF = dataDF.groupby([hue]).mean()
        alpha = 1
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


def gene_plot_conditions(X, condition: str, genes, ax, hue=None, unique=None):
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


def geneSig_plot_cells(X, comps, hue: str, ax, unique=None, average=False, kde=False):
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
        geneSigDF = geneSigDF.groupby([hue]).mean()
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
