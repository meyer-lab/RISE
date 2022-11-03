"""
Applying ULTRA to CoH
"""
import pandas as pd
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..CoHimport import CoH_df, CoH_xarray, cell_types
from ..tensor import optimal_seed, cell_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from ..regression import CoH_LogReg_plot
 
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    cond = ['Untreated', 'IFNg-50ng', 'IL10-50ng', 'IL4-50ng', 'IL2-50ng', 'IL6-50ng']; numCell = 5
    rank = 3; n_cluster = 9 ; seed = 1
    cohXA, cohDF, celltypeXA = CoH_xarray(numCell,cond,allmarkers=True)
    _, _, fit = optimal_seed(seed, cohXA, rank=rank, n_cluster=n_cluster)

    fac = fit[0]

    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.norm_NK(), color='k')
    ax[0].set(xlabel="Cluster", ylabel="Cell Abundance", xticks = np.arange(1, n_cluster + 1))

    # CP factors
    facXA = fac.get_factors_xarray(cohXA)
    cmap = sns.diverging_palette(240, 10, n=9)
    plotFactors(facXA, n_cluster,cmap, ax)

    # Soft Clustering Assignments
    cluster_type(cohXA, fac, celltypeXA, ax[5])

    # Logistic regression coefficients
    CoH_LogReg_plot(ax[6], fac, cohXA, rank)

    return f

def plotFactors(XA, n_cluster, cmap, ax):
    """Plots the individual factors for clusters, markers, and patients"""
    for i, key in enumerate(XA):
        if i < 4:
            data = XA[key]    
            if i == 0:
                sns.heatmap(
                data=data,
                xticklabels=data.coords[data.dims[1]].values,
                yticklabels=[f"Clst:{i}" for i in np.arange(1, n_cluster + 1)],
                ax=ax[i + 1], cmap=cmap, vmin=-1, vmax=1)
                
            else:
                sns.heatmap(
                data=data,
                xticklabels=data.coords[data.dims[1]].values,
                yticklabels=data.coords[data.dims[0]].values,
                ax=ax[i + 1], cmap=cmap, vmin=-1, vmax=1)
                
    
def cluster_type(flowXA, fac, typeXA, ax):
    """Soft assignments for clusters to true cell type"""
    # Solves for cell assignments shape [Cell, Cell Prob (Clust), Time, Dose, Ligand]
    resps = cell_assignment(flowXA.to_numpy(), fac) 
    # Normalizes each responsibility for every cell to equal 1 for every condition
    resps = resps / np.reshape(np.sum(resps, axis=1), (-1, 1, flowXA.shape[2], flowXA.shape[3], flowXA.shape[4]))
    
    # Finding hard clustering assignment for each cell 
    tensor_pred = np.argmax(resps, axis=1)+1

    # Comparing between predicted and actual assignments of all cells
    print("Rand_Score:", adjusted_rand_score(np.ravel(typeXA.to_numpy()),np.ravel(tensor_pred.astype(int))))
    # 1.0 is a perfect match for rand_score
    
    type_tensor = typeXA.to_numpy()
    clustDF = pd.DataFrame()
    # Iteratives over each cell type and cluster, sums the normalized responsibilites of partial clustering
    for i in range(len(cell_types)):
        truecell = type_tensor == i # Find where cell type is equal 
        truecell_index = np.argwhere(truecell == True) # Finds indices where true
        totalresps = np.sum(resps[truecell_index[:,0], :, truecell_index[:,1], truecell_index[:,2]],axis=0) / np.count_nonzero(truecell)
        for j in range(resps.shape[1]):
            clustDF = pd.concat([clustDF, pd.DataFrame({"Cluster": [j + 1], "Cell Type": [cell_types[i]], "Total Resp": np.asarray(totalresps[j])})])

    clustDF = clustDF.reset_index(drop=True)
    clustDF = clustDF.pivot(index="Cell Type", columns="Cluster", values="Total Resp")
    clustDF = clustDF.div(clustDF.sum(axis=0))
    assert np.isfinite(clustDF.to_numpy().all())
    
    cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
    sns.heatmap(data=clustDF, ax=ax, cmap=cmap)
