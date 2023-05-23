"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import pandas as pd
import math
from .common import subplotLabel, getSetup, flattenData
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 4))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    #ranks = np.arange(1, 31, step=10)
    
    ranks = [5, 15, 25, 35]
    #PCA_look(data, ranks, "NKG7", 0.1, ax[0:15])
    dim_red_var_drug(data, ranks, "Budesonide", ax[0])
    dim_red_var_drug(data, ranks, "Loteprednol etabonate", ax[1])
    dim_red_var_drug(data, ranks, "Betamethasone Valerate", ax[2])
    dim_red_var_drug(data, ranks, "Triamcinolone Acetonide", ax[3])

    dim_red_var_cell(data, ranks, "NKG7", 0.1, ax[4])
    dim_red_var_cell(data, ranks, "CD79A", 0.1, ax[5])
    dim_red_var_cell(data, ranks, "CD3D", 0.1, ax[6])
    dim_red_var_cell(data, ranks, "LAD1", 0.1, ax[7])

    all_drug_dist(data, 30, ax[8])
    all_marker_dist(data, 30, 0.1, ax[9])

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
    dist_DF = pd.DataFrame()
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


        Pf2_pwise = centroid_dist(Pf2_cell) / np.mean(pdist(Pf2_all.tolist()))
        PC_pwise = centroid_dist(PC_cell) / np.mean(pdist(PC_all.tolist()))
        dist_DF = pd.concat([dist_DF, pd.DataFrame({"Rank": [rank], "Norm Centroid Distance": Pf2_pwise, "Method": "PARAFAC2"})])
        dist_DF = pd.concat([dist_DF, pd.DataFrame({"Rank": [rank], "Norm Centroid Distance": PC_pwise, "Method": "PCA"})])
    
    var_DF = var_DF.reset_index(drop=True)
    dist_DF = dist_DF.reset_index(drop=True)
    sns.lineplot(data=dist_DF, x="Rank", y="Norm Centroid Distance", hue="Method", ax=ax)
    ax.set(title=marker)


def PCA_look(data, ranks, marker, cutoff, ax):
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
        projDF["Cell"] = dataDF.Cell.values
        Pf2_all = projDF.values[:, 0:-2]
        Pf2_cell = projDF.loc[dataDF.Cell == "Marker Positive"].values[:, 0:-2]

    for i in np.arange(0, rank-1):
        sns.scatterplot(data=projDF, x="Cmp. " + str(i+1), y="Cmp. " + str(i+2), hue="Cell", ax=ax[i])


def inter_clust_dist(clustPoints, otherPoints):
    """Calculates mean distance between clust points and all others"""
    dists = []
    clustVec = np.tile(clustPoints, (otherPoints.shape[0], 1))
    otherPointsVec = np.repeat(otherPoints, np.repeat(clustPoints.shape[0], otherPoints.shape[0], 0), axis=0)
    dists = np.zeros(clustVec.shape[0])
    for i in range(clustVec.shape[0]):
        dists[i] = math.dist(clustVec[i, :], otherPointsVec[i, :])
    return np.mean(dists)


def centroid_dist(points):
    """Calculates mean distance between clust points and all others"""
    centroid = np.mean(points, axis=1)
    dist_vecs = points - centroid[0]
    row_norms = np.linalg.norm(np.array(dist_vecs, dtype=np.float64), axis=1)
    return np.mean(row_norms)


def all_drug_dist(data, rank, ax):
    """Calculates mean distance between clust points and all others"""
    var_DF = pd.DataFrame()

    _, factors, projs, _ = parafac2_nd(
    data,
    rank=rank,
    random_state=1,
    verbose=True,
    )
    _, projDF, _ = flattenData(data, factors, projs)
    Pf2_all = projDF.values[:, 0:-1]

    pc = PCA(n_components=rank)
    PC_all = pc.fit_transform(data.unfold())

    for drug in projDF.Drug.unique():
        Pf2_drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]
        PC_drug = PC_all[projDF.Drug == drug]
            
        Pf2_var = np.sum(np.var(Pf2_drug, axis=0)) / np.sum(np.var(Pf2_all, axis=0))
        PC_var = np.sum(np.var(PC_drug, axis=0)) / np.sum(np.var(PC_all, axis=0))
        var_DF = pd.concat([var_DF, pd.DataFrame({"Drug": [drug], "% Total Variance": Pf2_var, "Method": "PARAFAC2"})])
        var_DF = pd.concat([var_DF, pd.DataFrame({"Drug": [drug], "% Total Variance": PC_var, "Method": "PCA"})])
    
    var_DF = var_DF.reset_index(drop=True)
    sns.swarmplot(data=var_DF, x="Method", y="% Total Variance", size=3, ax=ax)
    ax.set(title="Rank = " + str(rank))


def all_marker_dist(data, rank, cutoff, ax):
    """Calculates mean distance between clust points and all others"""
    dist_DF = pd.DataFrame()

    _, factors, projs, _ = parafac2_nd(
    data,
    rank=rank,
    random_state=1,
    verbose=True,
    )
    dataDF_init, projDF, _ = flattenData(data, factors, projs)
    Pf2_all = projDF.values[:, 0:-1]

    pc = PCA(n_components=rank)
    PC_all = pc.fit_transform(data.unfold())

    markers = [item for value in marker_genes.values() for item in (value if isinstance(value, list) else [value])]

    for marker in markers:
        if marker in dataDF_init.columns:
            dataDF, projDF, _ = flattenData(data, factors, projs)
            dataDF["Cell"] = "Other"
            dataDF.loc[dataDF[marker] > cutoff, "Cell"] = "Marker Positive"
            Pf2_all = projDF.values[:, 0:-1]
            Pf2_cell = projDF.loc[dataDF.Cell == "Marker Positive"].values[:, 0:-1]
            
            pc = PCA(n_components=rank)
            PC_all = pc.fit_transform(data.unfold())
            PC_cell = PC_all[dataDF.Cell == "Marker Positive"]

            Pf2_pwise = centroid_dist(Pf2_cell) / np.mean(pdist(Pf2_all.tolist()))
            PC_pwise = centroid_dist(PC_cell) / np.mean(pdist(PC_all.tolist()))
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Marker": [marker], "Norm Centroid Distance": Pf2_pwise, "Method": "PARAFAC2"})])
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Marker": [marker], "Norm Centroid Distance": PC_pwise, "Method": "PCA"})])

    dist_DF = dist_DF.reset_index(drop=True)
    print(dist_DF)
    sns.swarmplot(data=dist_DF, x="Method", y="Norm Centroid Distance", size=3, ax=ax)
    ax.set(title="Rank = " + str(rank))


marker_genes = {
    'Monocytes': [
        'CD14',
        'CD33',
        'LYZ',
        'LGALS3',
        'CSF1R',
        'ITGAX',
        'HLA-DRB1'],
    'Dendritic Cells': [
        'LAD1',
        'LAMP3',
        'TSPAN13',
        'CLIC2',
        'FLT3'],
    'B-cells': [
        'MS4A1',
        'CD19',
        'CD79A'],
    'T-helpers': [
        'TNF',
        'TNFRSF18',
        'IFNG',
        'IL2RA',
        'BATF'],
    'T cells': [
        'CD27',
        'CD69',
        'CD2',
        'CD3D',
        'CXCR3',
        'CCL5',
        'IL7R',
        'CXCL8',
        'GZMK'],
    'Natural Killers': [
        'NKG7',
        'GNLY',
        'PRF1',
        'FCGR3A',
        'NCAM1',
        'TYROBP'],
    'CD8': [
        'CD8A',
        'CD8B']}