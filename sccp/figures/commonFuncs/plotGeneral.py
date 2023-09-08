import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from ...crossVal import CrossVal
from ...decomposition import R2X


def plotR2X(data, rank, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = R2X(data, rank)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["Fit: Pf2", "Fit: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["|", "_"]

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

    
def plotGenePerCellType(genes, dataDF, ax):
    """Plots average gene expression across cell types for all conditions"""
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    df = data.groupby(["Condition", "Cell Type", "Gene"]).mean()
    
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"}).reset_index()
    print(df)
    # df = df.loc[df["Cell Type"] == "B Cells"]
    drug = "Dexrazoxane HCl (ICRF-187, ADR-529)"
    drugA = "Dex HCl"
    # df["Condition"]= df["Condition"].where(df["Condition"] == drug, drugA).where(df["Condition"] != drug, "Oth")
    # df["Condition"]= df["Condition"].where(df["Condition"] == drug, "Oth")
    

    df["Condition"] = np.where(df["Condition"].isin([drug]), df["Condition"], "Other")
    df = df.replace({"Condition": {drug: drugA}})
    
    print(np.unique(df["Condition"]))
    print(df)
    # df["Condition"]= df["Condition"].where(df["Condition"] == drug, drug).where(df["Condition"] < 60, 'Pass')
    # df["Condition"]= df['score'].where(df['score']  60, "Dexrazoxane HCl (ICRF-187, ADR-529").where(df['score'] < 60, 'Pass')
    sns.stripplot(data=df, x="Gene", y="Average Gene Expression For Drugs", hue="Cell Type", style = "Condition", dodge=True, jitter=False, ax=ax)
    

 