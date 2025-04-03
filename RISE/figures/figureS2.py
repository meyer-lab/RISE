"""
Figure S2
"""

import anndata
import numpy as np
import pacmap
import pandas as pd
import scanpy as sc
import scib
from sklearn.decomposition import PCA

from .common import getSetup, subplotLabel

# from .commonFuncs.plotGeneral import plot_r2x
from .commonFuncs.plotPaCMAP import plot_labels_pacmap

# import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    # ranks = [
    #     1,
    #     5,
    #     10,
    #     15,
    #     20,
    #     25,
    #     30,
    # ]
    # plot_r2x(X, ranks, ax[0])

    for i in range(3):
        ax[i].set(xticks=[0, 5, 10, 15, 20, 25, 30])

    drugs = ["Triamcinolone Acetonide", "Budesonide"]
    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 1], drug, cmap="Set1")
        ax[i + 1].set(title="Pf2-Based Decomposition")

    pc = PCA(n_components=20)
    pcaPoints = pc.fit_transform(np.asarray(X.X - X.var["means"].values))
    X.obsm["X_pf2_PaCMAP"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, drug in enumerate(drugs):
        plot_labels_pacmap(X, "Condition", ax[i + 3], drug, cmap="Set1")
        ax[i + 3].set(title="PCA-Based Decomposition")

    # pf2_batch_df = batch_correction_metrics(X, "projections")
    # pf2_batch_df["Fit"] = "Pf2"

    # X.obsm["pca"] = pcaPoints
    # pca_batch_df = batch_correction_metrics(X, "pca")
    # pca_batch_df["Fit"] = "PCA"

    # batch_df = pd.concat([pf2_batch_df, pca_batch_df])
    # sns.barplot(data=batch_df, x="Value", y="Metric", hue="Fit", ax=ax[5])

    return f


def batch_correction_metrics(X: anndata.AnnData, embed: str):
    """Returns SCIB metrics for batch correction and biological conservation"""
    clisi = scib.me.clisi_graph(
        X.to_memory(), label_key="Cell Type", type_="embed", use_rep=embed
    )
    ils = scib.me.isolated_labels_asw(
        X.to_memory(), batch_key="Condition", label_key="Cell Type", embed=embed
    )
    silh = scib.me.silhouette(X, label_key="Cell Type", embed=embed)
    sb = scib.me.silhouette_batch(
        X, batch_key="Condition", label_key="Cell Type", embed=embed
    )
    ilisi = scib.me.ilisi_graph(
        X.to_memory(), batch_key="Condition", type_="embed", use_rep=embed
    )
    sc.pp.neighbors(X, use_rep=embed)
    ilf = scib.me.isolated_labels_f1(
        X.to_memory(), batch_key="Condition", label_key="Cell Type", embed=embed
    )
    gc = scib.me.graph_connectivity(X, label_key="Cell Type")
    scib.me.cluster_optimal_resolution(X, cluster_key="leiden", label_key="Cell Type")
    ari = scib.me.ari(X, cluster_key="leiden", label_key="Cell Type")
    nmi = scib.me.nmi(X, cluster_key="leiden", label_key="Cell Type")

    metric_df = pd.DataFrame(
        [
            {
                "SB": sb,
                "GC": gc,
                "iLisi": ilisi,
                "cLisi": clisi,
                "ILS": ils,
                "S": silh,
                "ARI": ari,
                "NMI": nmi,
                "ILF": ilf,
            }
        ]
    )

    metric_df = metric_df.reset_index(drop=True)

    metric_df = pd.melt(
        metric_df,
        value_vars=["SB", "GC", "iLisi", "cLisi", "ILS", "ARI", "S", "NMI", "ILF"],
    ).rename(columns={"variable": "Metric", "value": "Value"})

    return metric_df
