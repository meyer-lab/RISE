import numpy as np
import anndata
from pacmap import PaCMAP
from parafac2 import parafac2_nd
from .imports import import_citeseq, import_lupus, import_thomson


def pf2(
    X: anndata.AnnData,
    condition_name: str,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
):
    # TensorFy
    # Sort so that the concatenation matches up later
    sort_idx = np.argsort(X.obs_vector(condition_name))
    X = X[sort_idx, :]

    # Get the indices for subsetting the data
    sgUnique, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    # We are going to center as we make the matrices
    means = np.mean(X.X, axis=0)

    X_pf = [X[sgIndex == sgi, :].X.toarray() - means for sgi in range(len(sgUnique))]

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
        X.obsm["embedding"] = PaCMAP(random_state=random_state).fit_transform(
            np.concatenate(projs, axis=0)
        )

    return X


def runAndSavePf2():
    """Runs the analysis and saves the cache files."""
    X = import_citeseq()
    X = pf2(X, "Condition", 80)
    X.write("CITEseq_analyzed_80comps.h5ad")

    X = import_lupus()
    X = pf2(X, "Condition", 40)
    X.write("Lupus_analyzed_40comps.h5ad")

    X = import_thomson()
    X = pf2(X, "Condition", 30)
    X.write("Thomson_analyzed_30comps.h5ad")
