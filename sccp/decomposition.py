import numpy as np
import numpy as np
from .parafac2 import parafac2_nd
from tensorly.decomposition._parafac2 import (
    _parafac2_reconstruction_error, _project_tensor_slices)
from tensorly.tenalg import khatri_rao
from sklearn.decomposition import PCA
import tlviz


def plotR2X_CC(tensor, rank, ax1, ax2):
    """Creates R2X plot for parafac2 tensor decomposition"""
    rank_vec = np.arange(1, rank + 1)

    flattenon = 2
    flatData = np.reshape(np.moveaxis(tensor, flattenon, 0), (tensor.shape[flattenon], -1))

    pf2_error = np.empty(len(rank_vec))
    pca_error = pf2_error.copy()
    core_consist = pf2_error.copy()

    # Collect Pf2 results
    for i in range(len(rank_vec)):
        weights, factors, projs = parafac2_nd(
                tensor,
                rank=i+1,
                verbose=True
            )
        
        # Core consistency 
        tensor_3d = np.reshape(tensor, (-1, tensor.shape[-2], tensor.shape[-1]))
        projected_tensor = _project_tensor_slices(tensor_3d, projs)
        projected_tensor_nD = np.reshape(
            projected_tensor, (*tensor.shape[0:-2], i+1, tensor.shape[-1])
        )
        
        core_consist[i] = tlviz.model_evaluation.core_consistency((weights, factors), projected_tensor_nD, normalised=True)
        
        if len(tensor) > 3:
            projs = np.reshape(projs, (-1, tensor.shape[-2], i+1))
            factors = [khatri_rao(factors[:-2]), factors[-2], factors[-1]]

        # R2X Pf2
        pf2_error[i] = 1 - np.square(_parafac2_reconstruction_error(
            tensor, (weights, factors, projs))) / np.square(np.linalg.norm(tensor))


    # Collect the PCA results
    pc = PCA(n_components=rank)
    pc.fit(flatData)

    pca_error = np.cumsum(pc.explained_variance_ratio_)
    total_error = np.vstack((pf2_error, pca_error))  

    name_decomp = ["Pf2", "PCA"]
    mark = ["x", "o", "*"]

    for i in range(total_error.shape[0]):
        ax1.scatter(rank_vec, total_error[i, :], label=name_decomp[i], marker=mark[i], s=20.0)
    ax1.set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(0, 1.05)
    )
    ax1.legend()
    
    ax2.scatter(rank_vec, core_consist/100, s=20.0)
    ax2.set(
        ylabel="Core Consistency",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(0, 1.05)
    )
    
