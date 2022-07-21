"""
Calculating SSE, NK and factors for PopAlign scRNA-seq (Allowing NK to vary over rank and cluster)
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup, subplotLabel
from gmm.scImport import ThompsonDrugXA, gene_import
from gmm.tensor import minimize_func, tensorGMM_CV
import scipy.cluster.hierarchy as sch


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
    maximizedNK, optCP, _, x, _, _ = minimize_func(drugXA, rank=rank, n_cluster=clust, nk_rearrange=True)
    print("LogLik", x)
    
    cmpCol = [f"Fac. {i}" for i in np.arange(1, fac + 1)]
    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]
    coords = {"Cluster": clustArray, "Factor": cmpCol, "Drug": drugXA.coords["Drug"]}
    maximizedFactors = [
        pd.DataFrame(optCP.factors[ii], columns=rankCol, index=coords[key])
        for ii, key in enumerate(coords)
    ]
    maximizedFactors[2] = reorder_table(maximizedFactors[2])

    NK_DF = pd.DataFrame(data=maximizedNK, columns=rankCol, index=clustArray)
    sns.heatmap(data=NK_DF, ax=ax[1])
    
    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 2])

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
