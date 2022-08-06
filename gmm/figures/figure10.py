"""
Calculating SSE, NK and factors for PopAlign scRNA-seq (Allowing NK to vary over rank and cluster)
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup
from gmm.scImport import ThompsonDrugXA
from gmm.tensor import minimize_func
import scipy.cluster.hierarchy as sch


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    drugXA, fac_vector, sse = ThompsonDrugXA()
    ax[0].plot(fac_vector, sse, "r")
    xlabel = "Number of Components"
    ylabel = "SSE"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    rank = 3
    clust = 3
    maximizedNK, optCP, _, x, _, _ = minimize_func(drugXA, rank=rank, n_cluster=clust, nk_rearrange=True)
    print("LogLik", x)
    
    cmpCol = [f"Fac. {i}" for i in fac_vector]
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

    return f


def reorder_table(df):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="right")["leaves"]

    return df.iloc[index, :]
