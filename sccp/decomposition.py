import numpy as np
from sklearn.decomposition import PCA
import tensorly as tl
from .parafac2 import parafac2_nd


def plotR2X(tensor, rank, ax1):
    """Creates R2X plot for parafac2 tensor decomposition"""
    rank_vec = np.arange(1, rank + 1)

    pf2_error = np.empty(len(rank_vec))

    # Collect Pf2 results
    for i in range(len(rank_vec)):
        _, _, _, pf2_error[i] = parafac2_nd(tensor, rank=i + 1, verbose=True)

    ax1.scatter(rank_vec, pf2_error, label="Pf2", marker="x", s=20.0)

    # Collect the PCA results
    pc = PCA(n_components=rank)
    pc.fit(tl.unfold(tensor, tensor.ndim - 1))
    pca_error = np.cumsum(pc.explained_variance_ratio_)

    ax1.scatter(
        rank_vec,
        pca_error,
        label="PCA",
        marker="o",
        s=20.0,
    )
    ax1.set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-0.05, np.max(np.append(pf2_error, pca_error) + 0.05))
    )
    ax1.legend()
