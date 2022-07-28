"""
Calculating SSE, NK and factors for PopAlign scRNA-seq 
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.scImport import ThompsonDrugXA, gene_import
from gmm.tensor import minimize_func, tensorGMM_CV
import scipy.cluster.hierarchy as sch


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # ax[5].axis("off")

    # geneDF = gene_import(offset=1.1,filter=True)

    num = 290
    fac = 3
    drugXA, fac_vector, sse = ThompsonDrugXA(numCells=num, rank=fac, maxit=2000, runFacts=False)
    ax[0].plot(fac_vector, sse, "r")
    xlabel = "Number of Components"
    ylabel = "SSE"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    rank = 3
    clust = 3
    maximizedNK, optCP, _, x, _, _ = minimize_func(drugXA, rank=rank, n_cluster=clust, nk_rearrange=False)
    print("LogLik", x)

    ax[1].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[1].set(xlabel=xlabel, ylabel=ylabel)

    cmpCol = [f"Fac. {i}" for i in np.arange(1, fac + 1)]
    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]
    coords = {"Cluster": clustArray, "Factor": cmpCol, "Drug": drugXA.coords["Drug"]}
    maximizedFactors = [
        pd.DataFrame(optCP.factors[ii], columns=rankCol, index=coords[key])
        for ii, key in enumerate(coords)]
    maximizedFactors[2] = reorder_table(maximizedFactors[2])

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 2])
    drug_gene_plot(maximizedFactors, "Alprostadil", fac, ax[5])
    # ranknumb = np.arange(2, 6)
    # n_cluster = np.arange(2, 6)

    # maxloglikDFcv = pd.DataFrame()
    # for i in range(len(ranknumb)):
    #     row = pd.DataFrame()
    #     row["Rank"] = ["Rank:" + str(ranknumb[i])]
    #     for j in range(len(n_cluster)):
    #         loglik = tensorGMM_CV(drugXA, numFolds=3, numClusters=n_cluster[j], numRank=ranknumb[i])
    #         print("LogLik", loglik)
    #         row["Cluster:" + str(n_cluster[j])] = loglik

    #     maxloglikDFcv = pd.concat([maxloglikDFcv, row])

    # maxloglikDFcv = maxloglikDFcv.set_index("Rank")
    # sns.heatmap(data=maxloglikDFcv, ax=ax[6])
    # ax[6].set(title="Cross Validation")

    return f


def reorder_table(df):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="right")["leaves"]

    return df.iloc[index, :]


def drug_gene_plot(factors_frame, drug, fac, ax):
    """Plots genes most associated with factor which is most associated with a drug"""
    max_fac = factors_frame[2].max(axis=1).loc[drug]
    max_drug_comp = factors_frame[2].transpose()[factors_frame[2].transpose()[drug] == max_fac].index.values
    max_fac_val = factors_frame[1][max_drug_comp].max().to_frame().values[0][0]
    max_fac_comp = factors_frame[1][factors_frame[1][max_drug_comp[0]] == max_fac_val].index.values
    NNMF_Comp = float(max_fac_comp[0][5::])
    geneFactors = pd.read_csv(join(path_here, "data/NNMF_Facts/NNMF_" + str(fac) + "_Loadings.csv")).drop("Unnamed: 0", axis=1)
    isoFac = geneFactors.loc[geneFactors.Component == NNMF_Comp].drop("Component", axis=1).transpose()
    isoFac = isoFac.reset_index()
    isoFac.columns = ["Gene", max_fac_comp[0]]
    isoFac = isoFac.sort_values(by=max_fac_comp[0]).tail(10)
    sns.barplot(data=isoFac, x="Gene", y=max_fac_comp[0], ax=ax, color='k')
    ax.set(title="Genes Upregulated by " + drug)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
