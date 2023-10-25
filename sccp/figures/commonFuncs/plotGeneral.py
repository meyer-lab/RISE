import numpy as np
import pandas as pd
import seaborn as sns
from ...crossVal import CrossVal
from ...decomposition import R2X
import anndata


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
