import numpy as np
from tqdm import tqdm
import anndata
import tensorly as tl
import cupy as cp
import scipy.sparse as sps
from scipy.sparse.linalg import svds
from pacmap import PaCMAP
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, CPTensor
from scipy.optimize import linear_sum_assignment
from tlviz.factor_tools import factor_match_score as fms, degeneracy_score


def cwSNR(
    X: anndata.AnnData,
) -> tuple[np.ndarray, float]:
    """Calculate the columnwise signal-to-noise ratio for each dataset and overall error."""
    a = X.uns["Pf2_A"] * X.uns["Pf2_weights"]
    SNR = a
    norm_overall = calc_total_norm(X)
    err_norm = 0.0

    # Get the indices for subsetting the data
    sgIndex = X.obs["condition_unique_idxs"]
    Xarr = sps.csr_array(X.X)
    W_proj = np.array(X.obsm["weighted_projections"])

    for i in range(X.uns["Pf2_A"].shape[0]):
        # Parafac2 to slice
        B_i = W_proj[sgIndex == i]
        slice = np.dot(B_i * a[i], np.array(X.varm["Pf2_C"]).T)

        X_condition_arr = Xarr[sgIndex == i] - X.var["means"].to_numpy()
        err_norm_here = float(np.linalg.norm(X_condition_arr - slice) ** 2.0)
        err_norm += err_norm_here

        SNR[i, :] /= err_norm_here

    return SNR, 1.0 - err_norm / norm_overall


def calc_total_norm(X: anndata.AnnData) -> float:
    """Calculate the total norm of the dataset, with centering"""
    Xarr = sps.csr_array(X.X)
    means = X.var["means"].to_numpy()

    # Deal with non-zero values first, by centering
    centered_nonzero = Xarr.data - means[Xarr.indices]
    centered_nonzero_norm = float(np.linalg.norm(centered_nonzero) ** 2.0)

    # Obtain non-zero counts for each column
    # Note that these are sorted, and no column should be empty
    unique, counts = np.unique(Xarr.indices, return_counts=True)
    assert np.all(np.diff(unique) == 1)

    num_zero = Xarr.shape[0] - counts
    assert num_zero.shape == means.shape
    zero_norm = np.sum(np.square(means) * num_zero)

    return zero_norm + centered_nonzero_norm


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
    parafac2_output = parafac2_nd(
        X,
        rank=rank,
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

    r2x_vec = np.empty(max_rank)

    for i in tqdm(range(len(r2x_vec)), total=len(r2x_vec)):
        parafac2_output = parafac2_nd(
            X,
            rank=i + 1,
        )

        X = store_pf2(X, parafac2_output)

        _, r2x_vec[i] = cwSNR(X)

    return r2x_vec


def _cmf_reconstruction_error(Xarr, sgIndex, means, factors: list, norm_X_sq: float):
    A, B, C = factors

    norm_sq_err = cp.array(norm_X_sq)
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        mat = cp.sparse.csr_matrix(Xarr[sgIndex == i])

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
    X: anndata.AnnData,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    verbose=True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    r"""The same interface as regular PARAFAC2."""
    # Index dataset to a list of conditions
    sgIndex = X.obs["condition_unique_idxs"]
    n_cond = np.amax(sgIndex) + 1

    Xarr = sps.csr_array(X.X)
    means = cp.array(X.var["means"].to_numpy())

    # Calculate the norm of the dataset
    norm_tensor = calc_total_norm(X)

    _, _, C = svds(Xarr, k=rank, return_singular_vectors=True)

    tl.set_backend("cupy")

    factors = [
        tl.ones((n_cond, rank), dtype=cp.float32),
        tl.eye(rank, dtype=cp.float32),
        cp.array(C.T),
    ]

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            Xarr, sgIndex, means, factors, norm_tensor
        )

        errs.append(tl.to_numpy((err / norm_tensor)))

        # Project tensor slices
        projected_X = tl.stack(projected_X)

        _, factors = parafac(
            projected_X,
            rank,
            n_iter_max=3,
            tol=None,  # type: ignore
            l2_reg=0.01,  # type: ignore
            init=(None, list(factors)),  # type: ignore
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    tl.set_backend("numpy")

    weights = np.ones(rank)
    factors = [f.get() for f in factors]
    projections = [p.get() for p in projections]  # type: ignore

    print(f"Degeneracy score: {degeneracy_score((weights, factors))}")

    return standardize_pf2(weights, factors, projections)


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
    indices = np.arange(0, X.shape[0])

    rng1 = np.random.default_rng(random_state)
    indices1 = rng1.choice(indices, size=X.shape[0], replace=True)

    rng2 = np.random.default_rng(random_state + 1)
    indices2 = rng2.choice(indices, size=X.shape[0], replace=True)

    X1 = X[indices1, :].to_memory()
    X2 = X[indices2, :].to_memory()

    fms_vec = np.empty(max_rank)

    for i in tqdm(range(len(fms_vec)), total=len(fms_vec)):
        parafac2_output1 = parafac2_nd(
            X1,
            rank=i + 1,
        )
        parafac2_output2 = parafac2_nd(
            X2,
            rank=i + 1,
        )

        X1cp = CPTensor((parafac2_output1[0], parafac2_output1[1]))
        X2cp = CPTensor((parafac2_output2[0], parafac2_output2[1]))

        fms_vec[i] = fms(X1cp, X2cp, consider_weights=True, skip_mode=None)

    return fms_vec