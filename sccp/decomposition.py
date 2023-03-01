import numpy as np
from sklearn.decomposition import PCA
import tensorly as tl
from .parafac2 import parafac2_nd


def plotR2X_CC(tensor, rank, ax1, ax2):
    """Creates R2X plot for parafac2 tensor decomposition"""
    rank_vec = np.arange(1, rank + 1)

    pf2_error = np.empty(len(rank_vec))
    core_consist = pf2_error.copy()

    # Collect Pf2 results
    for i in range(len(rank_vec)):
        _, _, _, pf2_error[i], core_consist[i] = parafac2_nd(
                tensor,
                rank=i+1,
                verbose=True
            )

    ax1.scatter(rank_vec, pf2_error, label="Pf2", marker="x", s=20.0)

    # Collect the PCA results
    pc = PCA(n_components=rank)
    pc.fit(tl.unfold(tensor, tensor.ndim - 2))

    ax1.scatter(rank_vec, np.cumsum(pc.explained_variance_ratio_), label="PCA", marker="o", s=20.0)
    ax1.set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-.05, 1.05)
    )
    ax1.legend()
    
    ax2.scatter(rank_vec, core_consist/100, s=20.0)
    ax2.set(
        ylabel="Core Consistency",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-.05, 1.05)
    )
    
