from typing import Sequence
import numpy as np
from tqdm import tqdm
import anndata
import tensorly as tl
import cupy as cp
import scipy.sparse as sps
from cupyx.scipy.sparse.linalg._norm import norm
from cupyx.scipy.sparse.linalg._eigen import svds
from pacmap import PaCMAP
from tensorly.decomposition import constrained_parafac
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, CPTensor
from scipy.optimize import linear_sum_assignment
from tlviz.factor_tools import factor_match_score as fms


def cwSNR(
    X: anndata.AnnData,
) -> tuple[np.ndarray, float]:
    """Calculate the columnwise signal-to-noise ratio for each dataset and overall error."""
    SNR = np.empty(X.uns["Pf2_A"].shape, dtype=float)
    norm_overall = 0.0
    err_norm = 0.0

    # Get the indices for subsetting the data
    sgIndex = X.obs["condition_unique_idxs"]
    Xarr = sps.csr_array(X.X)
    W_proj = np.array(X.obsm["weighted_projections"])

    for i in range(X.uns["Pf2_A"].shape[0]):

        # Parafac2 to slice
        a = X.uns["Pf2_A"][i] * X.uns["Pf2_weights"]
        B_i = W_proj[sgIndex == i]
        slice = np.dot(B_i * a, X.varm["Pf2_C"].T)

        X_condition_arr = Xarr[sgIndex == i] - X.var["means"].to_numpy()
        norm_overall += float(np.linalg.norm(X_condition_arr) ** 2.0)
        err_norm_here = float(np.linalg.norm(X_condition_arr - slice) ** 2.0)
        err_norm += err_norm_here

        SNR[i, :] = a**2.0
        SNR[i, :] /= err_norm_here

    return SNR, 1.0 - err_norm / norm_overall


def store_pf2(
    X: anndata.AnnData, parafac2_output: tuple[np.ndarray, list, list]
) -> anndata.AnnData:
    """Store the Pf2 results into the anndata object."""
    sgIndex = X.obs["condition_unique_idxs"]

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = parafac2_output[1]

    X.obsm["projections"] = np.zeros((X.shape[0], len(X.uns["Pf2_weights"])))
    for i, p in enumerate(parafac2_output[2]):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X


def pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
):
    X_pf = compress_tensor_slices(X)

    parafac2_output = parafac2_nd(
        X_pf,
        rank=rank,
        means=cp.array(X.var["means"].to_numpy())
    )

    X = store_pf2(X, parafac2_output)

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["embedding"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def pf2_r2x(
    X: anndata.AnnData,
    max_rank: int,
) -> np.ndarray:
    X = X.to_memory()
    X_pf = compress_tensor_slices(X)

    r2x_vec = np.empty(max_rank)

    for i in tqdm(range(len(r2x_vec)), total=len(r2x_vec)):
        parafac2_output = parafac2_nd(
            X_pf,
            rank=i + 1,
            means=cp.array(X.var["means"].to_numpy())
        )

        X = store_pf2(X, parafac2_output)

        _, r2x_vec[i] = cwSNR(X)

    return r2x_vec


def compress_tensor_slices(X: anndata.AnnData) -> list[cp.sparse.csr_matrix]:
    r"""Compress data with the randomized range finder for running PARAFAC2."""
    # Get the indices for subsetting the data
    sgIndex = X.obs["condition_unique_idxs"]
    n_cond = np.amax(sgIndex) + 1

    Xarr = sps.csr_array(X.X)
    X_pf = [cp.sparse.csr_matrix(Xarr[sgIndex == i]) for i in range(n_cond)]

    return X_pf


def _cmf_reconstruction_error(matrices: Sequence, factors: list, means, norm_X_sq):
    A, B, C = factors

    norm_sq_err = norm_X_sq.copy()
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        lhs = B @ (A[i] * C).T
        U, _, Vh = cp.linalg.svd(mat @ lhs.T - means @ lhs.T, full_matrices=False)
        proj = U @ Vh

        projections.append(proj)

        # Account for centering
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X.append(proj.T @ mat - centering)

        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * cp.trace(A[i][:, np.newaxis] * B.T @ projected_X[-1] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err, projections, projected_X


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    means: cp.ndarray,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    verbose=True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    r"""The same interface as regular PARAFAC2."""
    norm_tensor = cp.sum(cp.array([norm(xx) ** 2 for xx in X_in]))

    tl.set_backend("cupy")

    _, _, C = svds(X_in[0], k=rank, return_singular_vectors=True)

    factors = [
        tl.ones((len(X_in), rank)),
        tl.eye(rank),
        C.T,
    ]

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X_in, factors, means, norm_tensor
        )

        errs.append(tl.to_numpy((err / norm_tensor)))

        # Project tensor slices
        projected_X = tl.stack(projected_X)

        _, factors = constrained_parafac(
            projected_X,
            rank,
            n_iter_max=2,
            tol_outer=None,
            l2_reg={0: 0.01, 1: 0.01, 2: 0.01},
            init=(None, list(factors)),
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    tl.set_backend("numpy")

    weights = np.ones(rank)
    factors = [f.get() for f in factors]
    projections = [p.get() for p in projections]

    import tlviz

    print(
        f"Degeneracy score: {tlviz.factor_tools.degeneracy_score((weights, factors))}"
    )

    weights, factors, projections = standardize_pf2(weights, factors, projections)


    tucker_congruence_scores = np.ones(shape=(rank, rank))

    for factor in factors:
        tucker_congruence_scores *= tlviz.utils.normalise(factor).T @ tlviz.utils.normalise(factor)

    print(np.argwhere(tucker_congruence_scores < -0.6))

    return weights, factors, projections


def standardize_pf2(
    weights: np.ndarray, factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    weights = weights[gini_idx]

    weights, factors = cp_normalize(cp_flip_sign((weights, factors), mode=1))

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections


def pf2_fms(
    X: anndata.AnnData,
    max_rank: int,
    random_state=1,
) -> np.ndarray:
    # Get the indices for subsetting the data
    rng = np.random.default_rng(random_state)
    indices = rng.choice(2, size=X.shape[0])

    X1 = X[indices == 0, :].to_memory()
    X2 = X[indices == 1, :].to_memory()

    X1_pf = compress_tensor_slices(X1)
    X2_pf = compress_tensor_slices(X2)

    fms_vec = np.empty(max_rank)

    for i in tqdm(range(len(fms_vec)), total=len(fms_vec)):
        parafac2_output1 = parafac2_nd(
            X1_pf,
            rank=i + 1,
            means=cp.array(X1.var["means"].to_numpy())
        )
        parafac2_output2 = parafac2_nd(
            X2_pf,
            rank=i + 1,
            means=cp.array(X2.var["means"].to_numpy())
        )


        X1cp = CPTensor((parafac2_output1[0], parafac2_output1[1]))
        X2cp = CPTensor((parafac2_output2[0], parafac2_output2[1]))

        fms_vec[i] = fms(X1cp, X2cp, consider_weights=True, skip_mode=None)

    return fms_vec
