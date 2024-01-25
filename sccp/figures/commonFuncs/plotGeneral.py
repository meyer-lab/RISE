import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata
from matplotlib.axes import Axes

from ...factorization import pf2_r2x
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


def plotGenePerCellType(genes, adata, ax, cellType="Cell Type"):
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
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"})
    sns.boxplot(
        data=df,
        x="Gene",
        y="Average Gene Expression For Drugs",
        hue="Cell Type",
        ax=ax,
        fliersize=0,
    )


def plotGenePerCategCond(
    conds, categoryCond, gene, adata, ax, mean=True, cellType="Cell Type"
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

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression For Drugs",
        hue="Condition",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)


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


def heatmapGeneFactors(
    cmps: list, dataIn: anndata.AnnData, ax: Axes, geneAmount: int = 20
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    df = pd.DataFrame(
        data=dataIn.varm["Pf2_C"],
        index=dataIn.var_names,
        columns=range(1, dataIn.varm["Pf2_C"].shape[1] + 1),
    )
    df = df.reset_index(names="Gene")

    genes = np.array([])
    for cmp in cmps:
        sortDF = df.sort_values(by=cmp)
        top = sortDF.iloc[-geneAmount:, :].Gene.values
        bottom = sortDF.iloc[:geneAmount:, :].Gene.values
        genes = np.concatenate((genes, np.flip(top)))
        genes = np.concatenate((genes, bottom))

    heatmapDF = df.loc[df.Gene.isin(genes)][cmps + ["Gene"]].set_index("Gene")
    vmax = np.abs(heatmapDF.values).max()

    sns.heatmap(
        data=heatmapDF.transpose()[genes], ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax
    )


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
        sns.scatterplot(data=dataDF, x=genes[0], y=genes[1], hue=hue, ax=ax, alpha=5)
    else:
        sns.scatterplot(data=dataDF, x=genes[0], y=genes[1], ax=ax, alpha=0.2)


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


def plotGenePerCategStatus(conds, categoryCond, gene, adata, ax, obs, mean=True, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    adata = adata.to_memory()
    genesV = adata[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF[obs] = genesV.obs[obs].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(dataDF, id_vars=[obs, "Cell Type", "Condition"], value_vars=gene).rename(
        columns={"variable": "Gene", "value": "Value"}
    )
    if mean is True:
        df = df.groupby([obs, "Cell Type", "Gene", "Condition"], observed=False).mean()

    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    df[obs] = np.where(df[obs].isin(conds), df[obs], "Other")
    for i in conds:
        df = df.replace({obs: {i: categoryCond}})


    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue=obs,
        ax=ax,
        showfliers=False
    )
    ax.set(title=gene)





def plot2GenePerCategStatus(conds, categoryCond, gene1, gene2, adata, ax, obs, mean=True, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    adata = adata.to_memory()
    gene = [gene1, gene2]

    for i in range(2):
        genesV = adata[:, gene[i]].X.todense()
        print(genesV)
        dataDF = genesV.to_df()
        print(dataDF)
        dataDF = dataDF.subtract(genesV.var["means"].values)
        dataDF[obs] = genesV.obs[obs].values
        print(dataDF)
        dataDF["Condition"] = genesV.obs["Condition"].values
        dataDF["Cell Type"] = genesV.obs[cellType].values

        df = pd.melt(dataDF, id_vars=[obs, "Cell Type", "Condition"], value_vars=gene[i]).rename(
            columns={"variable": "Gene", "value": "Value"}
        )
        if mean is True:
            df = df.groupby([obs, "Cell Type", "Gene", "Condition"], observed=False).mean()

        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

        df[obs] = np.where(df[obs].isin(conds), df[obs], "Other")
        for j in conds:
            df = df.replace({obs: {j: categoryCond}})
        if i == 0:
            dfA = df.copy().drop(columns="Condition")
            dfA = dfA.groupby(["SLE_status", "Cell Type", "Gene"]).mean()

        else: 
            df = df.drop(columns="Condition")
            df = df.groupby(["SLE_status", "Cell Type", "Gene"]).mean()
            DF = pd.concat([dfA, df]).reset_index()

    df = DF.pivot(index=["SLE_status", "Cell Type"], columns="Gene", values="Average Gene Expression")


    sns.scatterplot(
        data=df,
        x=gene[0],
        y=gene[1],
        hue="Cell Type",
        ax=ax,
        style=obs
    )
    
    
def plot2GenePerCategStatusCellType(conds, categoryCond, gene1, gene2, adata, ax, obs, celltype, mean=True, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    adata = adata.to_memory()
    gene = [gene1, gene2]


    for i in range(2):
        genesV = adata[:, gene[i]]
        dataDF = pd.DataFrame(columns=gene[i], data=genesV.X.data)
        print(dataDF)
        a
        dataDF = genesV.to_df()
        dataDF = dataDF.subtract(genesV.var["means"].values)
        dataDF[obs] = genesV.obs[obs].values
        dataDF["Condition"] = genesV.obs["Condition"].values
        dataDF["Cell Type"] = genesV.obs[cellType].values
        df = pd.melt(dataDF, id_vars=["SLE_status", "Cell Type", "Condition"], value_vars=gene[i]).rename(
            columns={"variable": "Gene", "value": "Value"}
        )
        if mean is True:
            df = df.groupby(by=["SLE_status", "Cell Type", "Gene", "Condition"]).mean()
    

        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

        df[obs] = np.where(df[obs].isin(conds), df[obs], "Other")
  
        for j in conds:
            df = df.replace({obs: {j: categoryCond}})
        if i == 0:
            dfA = df.copy()

        else: 
            DF = pd.concat([dfA, df]).reset_index()

    
    df = DF.pivot(index=["SLE_status", "Cell Type", "Condition"], columns="Gene", values="Average Gene Expression").reset_index()
    
    
    print(df)
    df = df.loc[df["Cell Type"] == celltype].reset_index()
    
    df = df[df.isna().any(axis=1)]
    

    sns.scatterplot(
        data=df,
        x=gene[0],
        y=gene[1],
        hue="SLE_status",
        ax=ax,
    )
    
def plot2GenePerCategStatus2CellType(conds, categoryCond, gene1, gene2, adata, ax, obs, celltype1, celltype2, mean=True, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    adata = adata.to_memory()
    gene = [gene1, gene2]


    for i in range(2):
        genesV = adata[:, gene[i]]
        dataDF = genesV.to_df()
        dataDF = dataDF.subtract(genesV.var["means"].values)
        dataDF[obs] = genesV.obs[obs].values
        dataDF["Condition"] = genesV.obs["Condition"].values
        dataDF["Cell Type"] = genesV.obs[cellType].values

        df = pd.melt(dataDF, id_vars=[obs, "Cell Type", "Condition"], value_vars=gene[i]).rename(
            columns={"variable": "Gene", "value": "Value"}
        )
        if mean is True:
            df = df.groupby([obs, "Cell Type", "Gene", "Condition"], observed=False).mean()

        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

        df[obs] = np.where(df[obs].isin(conds), df[obs], "Other")
        for j in conds:
            df = df.replace({obs: {j: categoryCond}})
        if i == 0:
            dfA = df.copy()

        else: 
            df = pd.concat([dfA, df]).reset_index()

    
    print(df)
    df = df.pivot(index=["SLE_status", "Cell Type", "Condition"], columns="Gene", values="Average Gene Expression").reset_index()
    
  
    df1 = df.loc[df["Cell Type"] == celltype1]
    
    
    df2 = df.loc[df["Cell Type"] == celltype2]

    df = pd.concat([df1,df2])
    print(df)
    
    print(df.loc[df["Condition"] == "1004_1004:dmx_YE_7-13"])

    sns.scatterplot(
        data=df,
        x=gene[0],
        y=gene[1],
        hue="SLE_status",
        ax=ax,
    )
    