"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup, flattenData
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 4))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    ranks = np.arange(1, 31, step=10)
    #ranks = [5]
    dim_red_var_drug(data, ranks, "Budesonide", ax[0])
    dim_red_var_drug(data, ranks, "Loteprednol etabonate", ax[1])
    dim_red_var_drug(data, ranks, "Betamethasone Valerate", ax[2])
    dim_red_var_drug(data, ranks, "Triamcinolone Acetonide", ax[3])
    dim_red_var_cell(data, ranks, "NKG7", 0.1, ax[4])
    dim_red_var_cell(data, ranks, "CD79A", 0.1, ax[5])
    dim_red_var_cell(data, ranks, "CD3D", 0.1, ax[6])
    dim_red_var_cell(data, ranks, "LAD1", 0.1, ax[7])
  

    return f


def dim_red_var_drug(data, ranks, drug, ax):
    """Plots normalized variance for either a variable or for a group of cells"""
    var_DF = pd.DataFrame()
    for rank in ranks:
        _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
        )
        _, projDF, _ = flattenData(data, factors, projs)
        Pf2_all = projDF.values[:, 0:-1]
        Pf2_drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]
        
        pc = PCA(n_components=rank)
        PC_all = pc.fit_transform(data.unfold())
        PC_drug = PC_all[projDF.Drug == drug]
        
        Pf2_var = np.sum(np.var(Pf2_drug, axis=0)) / np.sum(np.var(Pf2_all, axis=0))
        PC_var = np.sum(np.var(PC_drug, axis=0)) / np.sum(np.var(PC_all, axis=0))
        var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": Pf2_var, "Method": "PARAFAC2"})])
        var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": PC_var, "Method": "PCA"})])
    
    var_DF = var_DF.reset_index(drop=True)
    sns.lineplot(data=var_DF, x="Rank", y="% Total Variance", hue="Method", ax=ax)
    ax.set(title=drug)


def dim_red_var_cell(data, ranks, marker, cutoff, ax):
    """Plots normalized variance for either a variable or for a group of cells"""
    var_DF = pd.DataFrame()
    for rank in ranks:
        _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
        )
        dataDF, projDF, _ = flattenData(data, factors, projs)
        dataDF["Cell"] = "Other"
        dataDF.loc[dataDF[marker] > cutoff, "Cell"] = "Marker Positive"
        Pf2_all = projDF.values[:, 0:-1]
        Pf2_cell = projDF.loc[dataDF.Cell == "Marker Positive"].values[:, 0:-1]
        
        pc = PCA(n_components=rank)
        PC_all = pc.fit_transform(data.unfold())
        PC_cell = PC_all[dataDF.Cell == "Marker Positive"]

        
        Pf2_var = np.sum(np.var(Pf2_cell, axis=0)) / np.sum(np.var(Pf2_all, axis=0))
        PC_var = np.sum(np.var(PC_cell, axis=0)) / np.sum(np.var(PC_all, axis=0))
        var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": Pf2_var, "Method": "PARAFAC2"})])
        var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": PC_var, "Method": "PCA"})])
    
    var_DF = var_DF.reset_index(drop=True)
    sns.lineplot(data=var_DF, x="Rank", y="% Total Variance", hue="Method", ax=ax)
    ax.set(title=marker)
