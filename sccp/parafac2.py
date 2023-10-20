import numpy as np
import anndata
from pacmap import PaCMAP
from parafac2 import parafac2_nd


def pf2(X: anndata.AnnData, condition_name: str, rank: int, random_state=1, doEmbedding=True):
    # TensorFy
    # Sort so that the concatenation matches up later
    sort_idx = np.argsort(X.obs_vector(condition_name))
    X = X[sort_idx, :]

    # Get the indices for subsetting the data
    sgUnique, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    X_pf = [X[sgIndex == sgi, :].X.toarray() for sgi in range(len(sgUnique))]

    weight, factors, projs, _ = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.obsm["projections"] = np.concatenate(projs, axis=0)
    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    if doEmbedding:
        X.obsm["embedding"] = PaCMAP(random_state=random_state).fit_transform(np.concatenate(projs, axis=0))

    return X
