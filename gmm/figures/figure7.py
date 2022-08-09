"""
Calculating SSE, NK and factors for PopAlign scRNA-seq 
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .common import getSetup
from gmm.scImport import ThompsonDrugXA
from gmm.tensor import minimize_func
import scipy.cluster.hierarchy as sch


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 12), (4, 3), multz={9: 2}, constrained_layout=False)

    nfac = 20
    drugXA, fac_vector, sse = ThompsonDrugXA(rank=nfac)

    ax[0].plot(fac_vector, sse, "r")
    xlabel = "Number of Components"
    ylabel = "SSE"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    rank = 4
    clust = 4
    fac, x, _ = minimize_func(drugXA, rank=rank, n_cluster=clust, nk_rearrange=True, maxiter=2000)
    print("LogLik", x)

    # ax[1].bar(np.arange(1, fac.nk.size + 1), fac.nk)
    # xlabel = "Cluster"
    # ylabel = "NK Value"
    # ax[1].set(xlabel=xlabel, ylabel=ylabel)

    facDF = fac.get_factors_dataframes(drugXA)
    facDF[2] = reorder_table(facDF[2], ax[9])

    for i in range(0, 3):
        sns.heatmap(data=facDF[i], vmin=0, ax=ax[i + 2])

    drug_gene_plot(facDF, "Budesonide", nfac, ax[5], max=True)
    drug_gene_plot(facDF, "Budesonide", nfac, ax[6], max=False)
    drug_gene_plot(facDF, "Dexrazoxane HCl (ICRF-187, ADR-529)", nfac, ax[7], max=True)
    drug_gene_plot(facDF, "Alprostadil", nfac, ax[8], max=True)
    plt.tight_layout()

    return f


def reorder_table(df, ax):
    """Reorder a table's rows using heirarchical clustering"""
    y = sch.linkage(df.to_numpy(), method="centroid")
    index = sch.dendrogram(y, orientation="top", ax=ax)["leaves"]
    ax.set_xticklabels(df.iloc[index, :].index.values, rotation=45, fontsize=6, ha="right")

    return df.iloc[index, :]


def drug_gene_plot(factors_frame: list, drug, fac: int, ax, max=True):
    """Plots genes most associated with factor which is most associated with a drug"""
    if max:
        max_fac = factors_frame[2].max(axis=1).loc[drug]
    else:
        max_fac = factors_frame[2].min(axis=1).loc[drug]
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

    if max:
        ax.set(title="Genes Upregulated by " + drug)
    else:
        ax.set(title="Genes Downregulated by " + drug)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
