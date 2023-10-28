import numpy as np
from sklearn.decomposition import PCA
from .parafac2 import parafac2_nd


def R2X(tensor, rank: int) -> tuple[list[float], np.ndarray]:
    """Creates R2X plot for parafac2 tensor decomposition"""
    rank_vec = np.arange(1, rank + 1)

    # Collect Pf2 results
    pf2_error = [parafac2_nd(tensor, rank=r)[3] for r in rank_vec]

    # Collect the PCA results
    pc = PCA(n_components=rank)
    pc.fit(np.concatenate(tensor, axis=0))
    pca_error = np.cumsum(pc.explained_variance_ratio_)

    return pf2_error, pca_error
