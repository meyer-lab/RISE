import os
from os.path import join
import numpy as np
from .parafac2 import _parafac2_rec_error, parafac2

path_here = os.path.dirname(os.path.dirname(__file__))

def plotR2X(tensor, rank, datatype, ax, runPf2=False):
    """Creates R2X plot for parafac2 tensor decomposition"""
    if runPf2 == True:
        rank_vec = np.arange(1, rank + 1)
        pf2_error = np.empty(len(rank_vec))
        for i in range(len(rank_vec)):
            weights, factors, projs = parafac2(
                    tensor.to_numpy(),
                    rank=rank,
                    n_iter_max=10,
                    nn_modes=(0, 1, 2),
                    verbose=True,
                )

            if len(tensor.shape) == 3:
                rearrange = False
            else:
                rearrange = True
            

            pf2_error[i] = _parafac2_rec_error(
                tensor.to_numpy(), [weights, factors, projs], rearrangeProjs=rearrange
            )

        if datatype == "IL2":
            np.save(join(path_here, "sccp/data/IL2_Pf2_Errors.npy"), pf2_error)
        elif datatype == "CoH":
            np.save(join(path_here, "sccp/data/CoH_Pf2_Errors.npy"), pf2_error)
        elif datatype == "scRNA":
            np.save(join(path_here, "sccp/data/scRNA_Pf2_Errors.npy"), pf2_error)
        elif datatype == "Synthetic":
            np.save(join(path_here, "sccp/data/Synthetic_Pf2_Errors.npy"), pf2_error)

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
        elif datatype == "Synthetic":
            pf2_error = np.load(
                join(path_here, "sccp/data/Synthetic_Pf2_Errors.npy"), allow_pickle=True
            )

        rank_vec = np.arange(1, len(pf2_error) + 1)
        ax.scatter(rank_vec, pf2_error, c="k", s=20.0)
        ax.set(
            title="R2X",
            ylabel="Variance Explained",
            xlabel="Number of Components",
            xticks=np.arange(0, rank + 1),
        )
