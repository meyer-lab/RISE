import os
import numpy as np
from os.path import join
import numpy as np
from .parafac2 import parafac2_nd
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.tenalg import khatri_rao
from tensorpack import Decomposition
from sklearn.decomposition import NMF

path_here = os.path.dirname(os.path.dirname(__file__))

def plotR2X(tensor, rank, datatype, ax, run_decomp=False, Inlclude_NNMF=False):
    """Creates R2X plot for parafac2 tensor decomposition"""
    rank_vec = np.arange(1, rank + 1)
    if run_decomp == True:
        pf2_error = np.empty(len(rank_vec))
        pca_error = pf2_error.copy()
        nnmf_error = pf2_error.copy()
        for i in range(len(rank_vec)):
            weights, factors, projs = parafac2_nd(
                    tensor,
                    rank=i+1,
                    verbose=True
                )

            if len(tensor) > 3:
                projs = np.reshape(projs, (-1, tensor.shape[-2], i+1))
                factors = [khatri_rao(factors[:-2]), factors[-2], factors[-1]]

            pf2_error[i] = 1 - _parafac2_reconstruction_error(
                tensor, (weights, factors, projs))/ np.linalg.norm(tensor) ** 2
            
            decomp = Decomposition(tensor, max_rr=i+1)
            decomp.perform_PCA(flattenon=2)
            
            pcaError = decomp.PCAR2X
            pca_error[i] = pcaError[-1]
            
            if Inlclude_NNMF == True:
                flat_tensor = np.reshape(np.moveaxis(tensor, 2, 0), (tensor.shape[2], -1))
                nnmf = NMF(n_components=i+1)
                nnmf.fit(flat_tensor)

        total_error = np.vstack((pf2_error, pca_error))  
        
        if Inlclude_NNMF == True:
            total_error = np.vstack((pf2_error, nnmf_error, pca_error)) 
        
        if datatype == "IL2":
            np.save(join(path_here, "sccp/data/IL2_Total_Errors.npy"), total_error)
        elif datatype == "CoH":
            np.save(join(path_here, "sccp/data/CoH_Total_Errors.npy"), total_error)
        elif datatype == "scRNA":
            np.save(join(path_here, "sccp/data/scRNA_Total_Errors.npy"), total_error)
        elif datatype == "Synthetic1":
            np.save(join(path_here, "sccp/data/Synthetic1_Total_Errors.npy"), total_error)
        elif datatype == "Synthetic2":
            np.save(join(path_here, "sccp/data/Synthetic2_Total_Errors.npy"), total_error)
        elif datatype == "Synthetic3":
            np.save(join(path_here, "sccp/data/Synthetic3_Total_Errors.npy"), total_error)
            
        

    else:
        if datatype == "IL2":
            total_error = np.load(
                join(path_here, "sccp/data/IL2_Total_Errors.npy"), allow_pickle=True
            )
        elif datatype == "CoH":
            total_error = np.load(
                join(path_here, "sccp/data/CoH_Total_Errors.npy"), allow_pickle=True
            )
        elif datatype == "scRNA":
            total_error = np.load(
                join(path_here, "sccp/data/scRNA_Total_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic1":
            total_error = np.load(
                join(path_here, "sccp/data/Synthetic1_Total_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic2":
            total_error = np.load(
                join(path_here, "sccp/data/Synthetic2_Total_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic3":
            total_error = np.load(
                join(path_here, "sccp/data/Synthetic3_Total_Errors.npy"), allow_pickle=True
            )

        name_decomp = ["Pf2", "PCA"]
        mark = ["x", "o", "*"]
        if Inlclude_NNMF == True:
                name_decomp = ["Pf2", "NNMF", "PCA"]
        for i in range(total_error.shape[0]):
            ax.scatter(rank_vec, total_error[i, :], label=name_decomp[i], marker=mark[i], s=20.0)
        ax.set(
            title="R2X",
            ylabel="Variance Explained",
            xlabel="Number of Components",
            xticks=np.arange(0, rank + 1),
            ylim=(0, 1.05)
        )
        ax.legend()

