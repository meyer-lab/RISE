"""
Investigating NK, covariance, and factors from tGMM for IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, markerslist, optimal_seed, cell_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    marks = ["Foxp3","CD25","pSTAT5"]
    zflowTensor, celltypeXA = smallDF(cellperexp)
    zflowTensor = zflowTensor.loc[marks,:,:,:,:]
    
    rank = 3
    n_cluster = 3

    optimalseed, min_loglik = optimal_seed(
        5, zflowTensor, rank=rank, n_cluster=n_cluster
    )
    print("Optimal Seed:", optimalseed)
    print("Min LogLik:", min_loglik)

    fac, x, _ = minimize_func(zflowTensor, rank=rank, n_cluster=n_cluster, seed=optimalseed)
   
    cluster_type(zflowTensor, fac, celltypeXA[1], ax[9], ax[10])
    
    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.norm_NK())
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    facXA = fac.get_factors_xarray(zflowTensor)

    for i, key in enumerate(facXA):
        data = facXA[key]
        sns.heatmap(
            data=data,
            xticklabels=data.coords[data.dims[1]].values,
            yticklabels=data.coords[data.dims[0]].values,
            vmin=0,
            ax=ax[i + 1],
        )

    # Covariance for different ranks
    for i in range(3):
        dff = pd.DataFrame(
            fac.covars[:, :, i] @ fac.covars[:, :, i].T,
            columns=marks,
            index=marks,
        )
        sns.heatmap(data=dff, ax=ax[i + 6])
        ax[i + 6].set(title="Covariance: Rank - " + str(i + 1))

    return f
    
def cluster_type(drugXA, fac, typeXA, ax1, ax2):
    """Solves for confusion matrix of predicted and actual cell type labels"""
    # Solves for cell assignments shape [Cell, Cell Prob (Clust), Time, Dose, Ligand]
    resps = cell_assignment(drugXA.to_numpy(), fac) 
    # Normalizes each responsibility for every cell to equal 1 for every condition
    resps = resps / np.reshape(np.sum(resps, axis=1), (-1, 1, drugXA.shape[2], drugXA.shape[3], drugXA.shape[4]))
    
    # Finding hard clustering assignment for each cell 
    tensor_pred = np.argmax(resps, axis=1)+1

    # Comparing between predicted and actual assignments of all cells
    print("Rand_Score:", adjusted_rand_score(np.ravel(typeXA.to_numpy()),np.ravel(tensor_pred.astype(int))))
    # 1.0 is a perfect match for rand_score
    
    confmatrix = confusion_matrix(np.ravel(typeXA.to_numpy()), np.ravel(tensor_pred.astype(int)))
    confDF = pd.DataFrame(data=confmatrix, index=["None", "Treg", "Thelper"], columns=[f"Clst. {i}" for i in np.arange(1, resps.shape[1] + 1)])
    sns.heatmap(data=confDF, ax=ax1, annot=True)
    ax1.set(title="HardClustering:ConfusionMatrix")
    
    type_tensor = typeXA.to_numpy()
    clustDF = pd.DataFrame()
    celltype_dict = ["None", "Treg", "Thelper"]
    # Iteratives over each cell type and cluster, sums the normalized responsibilites of partial clustering
    for i in range(len(celltype_dict)):
        truecell = type_tensor == i+1
        truecell_index = np.argwhere(truecell == True)
        totalresps = np.sum(resps[truecell_index[:,0], :, truecell_index[:,1], truecell_index[:,2], truecell_index[:,3]],axis=0)
        for j in range(resps.shape[1]):
            clustDF = pd.concat([clustDF, pd.DataFrame({"Cluster": [j + 1], "Cell Type": [celltype_dict[i]], "Total Resp": np.asarray(totalresps[j])})])

    clustDF = clustDF.reset_index(drop=True)
    clustDF = clustDF.pivot(index="Cell Type", columns="Cluster", values="Total Resp")
    clustDF = clustDF.div(clustDF.sum(axis=0))
    assert np.isfinite(clustDF.to_numpy().all())
    sns.heatmap(data=clustDF, ax=ax2, annot=True)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set(title="SoftClustering:CellPercentages")
                        
                        
        
        
            