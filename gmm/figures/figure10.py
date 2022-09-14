"""
Calculating SSE, NK and factors for PopAlign scRNA-seq (Allowing NK to vary over rank and cluster)
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup
from gmm.scImport import ThompsonDrugXA
from gmm.tensor import optimal_seed
import scipy.cluster.hierarchy as sch


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    drugXA, _, fac_vector, sse = ThompsonDrugXA()
    ax[0].plot(fac_vector, sse, "r")
    xlabel = "Number of Components"
    ylabel = "SSE"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    rank = 5
    clust = 3

    _, _, fit = optimal_seed(
        2, drugXA, rank=rank, n_cluster=clust, nk_rearrange=True
    )
    fac = fit[0]

    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]

    facXA = fac.get_factors_xarray(drugXA)

    NK_DF = pd.DataFrame(data=fac.nk, columns=rankCol, index=clustArray)
    sns.heatmap(data=NK_DF, ax=ax[1])

    for i, key in enumerate(facXA):
        if i < 3:
            data = facXA[key]
            sns.heatmap(
                data=data,
                xticklabels=data.coords[data.dims[1]].values,
                yticklabels=data.coords[data.dims[0]].values,
                ax=ax[i + 2])    
    return f


def reorder_table(df):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="right")["leaves"]

    return df.iloc[index, :]
