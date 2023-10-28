import os
from typing import Sequence
import numpy as np
import anndata
import tensorly as tl
from pacmap import PaCMAP
from .imports import import_citeseq, import_lupus, import_thomson
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.cp_tensor import cp_flip_sign
from tensorly.tenalg.svd import randomized_svd
from tensorly.decomposition import parafac2
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
from scipy.optimize import linear_sum_assignment


def cwSNR(
    matrices: Sequence,
    weights: np.ndarray,
    factors: list[np.ndarray],
    projections: list[np.ndarray],
):
    """Calculate the columnwise signal-to-noise ratio for each dataset."""
    SNR = np.empty((len(matrices), len(weights)), dtype=float)

    for i, mat in enumerate(matrices):
        xx = parafac2_to_slice((weights, factors, projections), i, validate=False)

        SNR[i, :] = (factors[0][i, :] * weights) ** 2.0
        SNR[i, :] /= np.linalg.norm(mat - xx) ** 2.0

    return SNR


def pf2(
    X: anndata.AnnData,
    condition_name: str,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
):
    # TensorFy
    # Get the indices for subsetting the data
    sgUnique, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    # We are going to center as we make the matrices
    means = np.mean(X.X, axis=0)

    X_pf = [X[sgIndex == sgi, :].X.toarray() - means for sgi in range(len(sgUnique))]

    # Quantify the variation in cross-products since this is an assumption of Pf2
    covs = np.stack([xx.T @ xx for xx in X_pf])
    cov_total = tl.norm(covs) ** 2
    cov_var = tl.norm(covs - np.mean(covs, axis=0)) ** 2

    weight, factors, projs, _ = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.uns["cov_ratio"] = cov_var / cov_total
    X.uns["cvSNR"] = cwSNR(X_pf, weight, factors, projs)

    X.obsm["projections"] = np.zeros((X.shape[0], rank))
    for i, p in enumerate(projs):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["embedding"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

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


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-7,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    norm_tensor = np.sum([np.linalg.norm(xx) ** 2 for xx in X_in])

    # Checks size of each experiment is bigger than rank
    for i in range(len(X_in)):
        assert np.shape(X_in[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X_in[0])[1] > rank

    # Assemble covariance matrix rather than concatenation
    # This saves memory and should be faster
    covM = X_in[0].T @ X_in[0]
    for i in range(1, len(X_in)):
        covM += X_in[i].T @ X_in[i]

    C = randomized_svd(covM, rank, random_state=rng, n_iter=4)[0]

    (w, f, p) = parafac2( # type: ignore
        X_in,
        rank,
        n_iter_max=n_iter_max,
        init=(None, [np.ones((len(X_in), rank)), np.eye(rank), C]), # type: ignore
        svd="truncated_svd",
        normalize_factors=True,
        tol=tol,
        nn_modes=(0,),
        random_state=rng,
        verbose=verbose,
        return_errors=False,
        n_iter_parafac=5,
        linesearch=True,
    )

    pf2_error = _parafac2_reconstruction_error(X_in, (w, f, p)) ** 2.0

    R2X = 1 - pf2_error / norm_tensor

    gini_idx = giniIndex(f[0])
    f = [f[:, gini_idx] for f in f]
    w = w[gini_idx]

    w, f = cp_flip_sign((w, f), mode=1)

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(f[1].T), maximize=True)
    f[1] = f[1][col_ind, :]
    p = [p[:, col_ind] for p in p]

    # Flip the sign based on B
    signn = np.sign(np.diag(f[1]))
    f[1] *= signn[:, np.newaxis]
    p = [p * signn for p in p]

    return w, f, p, R2X


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment"""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)
