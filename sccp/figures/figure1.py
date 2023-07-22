"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
from ..imports.scRNA import import_pancreas, import_pancreas_all
from ..parafac2 import parafac2_nd
import scib
import pandas as pd
import seaborn as sns
import warnings
import numpy as np 

warnings.filterwarnings("ignore")

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    methods=["bbknn", "scanorama", "harmony"]
    ranks = [2]
    pancreas_pf2 = import_pancreas(tensor=True)
    rank = 2
    _, factors, projs, _ = parafac2_nd(pancreas_pf2, rank=rank, random_state=1)
    pancreas = import_pancreas(tensor=False)
    flatProjs = np.concatenate(projs, axis=0)
    pancreas.obsm["Pf2"] = flatProjs

    compare_int_methods(methods, flatProjs, ax=ax[0:2])
    compare_ranks_methods(ranks, ax=ax[2:4])

    return f


def compare_int_methods(methods, projs, ax): 
    """Compares all sc methodologies using scib metrics"""
    metric_df = pd.DataFrame()
    pancreas = import_pancreas(tensor=False)
    asw = scib.me.isolated_labels_asw(pancreas, batch_key="batch", label_key="celltype", embed="X_pca")
    kBET = 1 - scib.me.kBET(pancreas, batch_key="batch", label_key="celltype", type_="full", embed="X_pca")
    metric_df = pd.concat([metric_df, pd.DataFrame({"Method": ["Unintegrated"], "ASW": asw, "kBET": kBET})])

    for method in methods:
        pancreas = import_pancreas(tensor=False, method="_" + method)
        asw = scib.me.isolated_labels_asw(pancreas, batch_key="batch", label_key="celltype", embed="X_pca")
        kBET = 1 - scib.me.kBET(pancreas, batch_key="batch", label_key="celltype", type_="full", embed="X_pca")
        metric_df = pd.concat([metric_df, pd.DataFrame({"Method": [method], "ASW": asw, "kBET": kBET})])

    pancreas_pf2 = import_pancreas(tensor=False)
    pancreas_pf2.obsm["Pf2"] = projs
    asw = scib.me.isolated_labels_asw(pancreas_pf2, batch_key="batch", label_key="celltype", embed="Pf2")
    kBET = 1 - scib.me.kBET(pancreas_pf2, batch_key="batch", label_key="celltype", type_="full", embed="Pf2")
    metric_df = pd.concat([metric_df, pd.DataFrame({"Method": ["PARAFAC2"], "ASW": asw, "kBET": kBET})])

    metric_df = metric_df.reset_index(drop=True)
    sns.barplot(data=metric_df, x="ASW", y="Method", ax=ax[0], color='k')
    sns.barplot(data=metric_df, x="kBET", y="Method", ax=ax[1], color='k')


def compare_ranks_methods(ranks, ax): 
    """Compares all sc methodologies using scib metrics"""
    metric_df = pd.DataFrame()
    pancreas_pf2 = import_pancreas(tensor=True)
    pancreas = import_pancreas(tensor=False)

    for rank in ranks:
        _, factors, projs, _ = parafac2_nd(pancreas_pf2, rank=rank, random_state=1)
        pancreas.obsm["Pf2"] = np.concatenate(projs, axis=0)
        asw = scib.me.isolated_labels_asw(pancreas, batch_key="batch", label_key="celltype", embed="Pf2")
        kBET = 1 - scib.me.kBET(pancreas, batch_key="batch", label_key="celltype", type_="full", embed="Pf2")
        metric_df = pd.concat([metric_df, pd.DataFrame({"Rank": [rank], "ASW": asw, "kBET": kBET})])

    metric_df = metric_df.reset_index(drop=True)
    sns.lineplot(data=metric_df, x="Rank", y="ASW", ax=ax[0])
    sns.lineplot(data=metric_df, x="Rank", y="kBET", ax=ax[1])


func_dict = {"scanvi": scib.ig.scanvi,
             "combat": scib.ig.combat,
             "scanorama": scib.ig.scanorama,
             "bbknn": scib.ig.bbknn}
