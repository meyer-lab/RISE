import os
import numpy as np
from os.path import join
import numpy as np
from .parafac2 import parafac2_nd
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from tensorly.tenalg import khatri_rao

path_here = os.path.dirname(os.path.dirname(__file__))

def plotR2X(tensor, rank, datatype, ax, runPf2=False):
    """Creates R2X plot for parafac2 tensor decomposition"""
    if runPf2 == True:
        rank_vec = np.arange(1, rank + 1)
        pf2_error = np.empty(len(rank_vec))
        for i in range(len(rank_vec)):
            weights, factors, projs = parafac2_nd(
                    tensor,
                    rank=rank,
                    verbose=True,
                )

            if len(tensor) > 3:
                projs = np.reshape(projs, (-1, tensor.shape[-2], rank))
                factors = [khatri_rao(factors[:-2]), factors[-2], factors[-1]]

            pf2_error[i] = 1 - _parafac2_reconstruction_error(
                tensor, (weights, factors, projs))/ np.linalg.norm(tensor) ** 2

        if datatype == "IL2":
            np.save(join(path_here, "sccp/data/IL2_Pf2_Errors.npy"), pf2_error)
        elif datatype == "CoH":
            np.save(join(path_here, "sccp/data/CoH_Pf2_Errors.npy"), pf2_error)
        elif datatype == "scRNA":
            np.save(join(path_here, "sccp/data/scRNA_Pf2_Errors.npy"), pf2_error)
        elif datatype == "Synthetic1":
            np.save(join(path_here, "sccp/data/Synthetic1_Pf2_Errors.npy"), pf2_error)
        elif datatype == "Synthetic2":
            np.save(join(path_here, "sccp/data/Synthetic2_Pf2_Errors.npy"), pf2_error)
        elif datatype == "Synthetic3":
            np.save(join(path_here, "sccp/data/Synthetic3_Pf2_Errors.npy"), pf2_error)

    else:
        if datatype == "IL2":
            pf2_error = np.load(
                join(path_here, "sccp/data/IL2_Pf2_Errors.npy"), allow_pickle=True
            )
        elif datatype == "CoH":
            pf2_error = np.load(
                join(path_here, "sccp/data/CoH_Pf2_Errors.npy"), allow_pickle=True
            )
        elif datatype == "scRNA":
            pf2_error = np.load(
                join(path_here, "sccp/data/scRNA_Pf2_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic1":
            pf2_error = np.load(
                join(path_here, "sccp/data/Synthetic1_Pf2_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic2":
            pf2_error = np.load(
                join(path_here, "sccp/data/Synthetic2_Pf2_Errors.npy"), allow_pickle=True
            )
        elif datatype == "Synthetic3":
            pf2_error = np.load(
                join(path_here, "sccp/data/Synthetic3_Pf2_Errors.npy"), allow_pickle=True
            )

            
        rank_vec = np.arange(1, len(pf2_error) + 1)
        ax.scatter(rank_vec, pf2_error, c="k", s=20.0)
        ax.set(
            title="R2X",
            ylabel="Variance Explained",
            xlabel="Number of Components",
            xticks=np.arange(0, rank + 1),
        )
