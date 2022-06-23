"""
This creates Figure 7.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.scImport import ThompsonDrugXA, gene_import
from gmm.tensor import minimize_func, tensorGMM_CV
import scipy.cluster.hierarchy as sch


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))
    
    ax[5].axis("off")

    num = 290
    fac = 20
    drugXA = ThompsonDrugXA(numCells=num, rank=fac, maxit=2000)
    
    rank = 5
    clust = 4
    maximizedNK, optCP, _, x, _, _ = minimize_func(drugXA, rank=rank, n_cluster=clust)
    print("LogLik", x)

    ax[0].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    cmpCol = [f"Fac. {i}" for i in np.arange(1, fac + 1)]
    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]
    coords = {"Cluster": clustArray, "Factor": cmpCol, "Drug": drugXA.coords["Drug"]}
    maximizedFactors = [pd.DataFrame(optCP.factors[ii], columns=rankCol, index=coords[key]) for ii, key in enumerate(coords)]
    maximizedFactors[2] = reorder_table(maximizedFactors[2])

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 1], cmap="Greens")

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
    # sns.heatmap(data=maxloglikDFcv, ax=ax[4])
    # ax[4].set(title="Cross Validation")

    return f

def reorder_table(df):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(y, orientation='right')['leaves']
        
    return df.iloc[index, :]
