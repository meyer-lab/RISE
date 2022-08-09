"""
Calculating SSE, NK and factors for PopAlign scRNA-seq (Allowing NK to vary over rank and cluster)
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.cp_tensor import cp_normalize
from .common import getSetup
from gmm.scImport import ThompsonDrugXA
from gmm.tensor import minimize_func, optimal_seed
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

    rank = 5
    clust = 3

    optimalseed, _ = optimal_seed(
        2, drugXA, rank=rank, n_cluster=clust, nk_rearrange=True
    )

    fac, x, _ = minimize_func(drugXA, rank=rank, n_cluster=clust, nk_rearrange=True, seed=optimalseed)
    print("LogLik", x)

    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]

    facDF = fac.get_factors_dataframes(drugXA)
    facDF[2] = reorder_table(facDF[2])

    NK_DF = pd.DataFrame(data=fac.nk, columns=rankCol, index=clustArray)
    sns.heatmap(data=NK_DF, ax=ax[1])
    
    for i in range(3):
        sns.heatmap(data=facDF[i], vmin=0, ax=ax[i + 2])

    return f


def reorder_table(df):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="right")["leaves"]

    return df.iloc[index, :]
