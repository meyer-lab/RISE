import numpy as np
from copy import deepcopy
from tqdm import tqdm
import anndata
import tensorly as tl
import cupy as cp
from cupyx.scipy import sparse as cupy_sparse
import scipy.sparse as sps
from cupyx.scipy.sparse.linalg import svds
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment
from tlviz.factor_tools import degeneracy_score


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


def reconstruction_error(
    factors: list, projections: list, projected_X: cp.ndarray, norm_X_sq: float
) -> float:
    """Calculate the reconstruction error from the factors and projected data."""
    A, B, C = factors
    CtC = C.T @ C

    norm_sq_err = cp.array(norm_X_sq)

    for i, proj in enumerate(projections):
        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * cp.trace(A[i][:, np.newaxis] * B.T @ projected_X[i] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return float(cp.asnumpy(norm_sq_err))


def project_data(
    Xarr: sps.csr_array, sgIndex: np.ndarray, means, factors: list
) -> tuple[cp.ndarray, cp.ndarray]:
    A, B, C = factors

    projections = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0]))

    for i in range(np.amax(sgIndex) + 1):
        # Prepare CuPy matrix
        mat = cupy_sparse.csr_matrix(Xarr[sgIndex == i], dtype=cp.float32)

        lhs = cp.array(B @ (A[i] * C).T, dtype=cp.float32)
        U, _, Vh = cp.linalg.svd(mat @ lhs.T - means @ lhs.T, full_matrices=False)
        proj = U @ Vh

        projections.append(proj)

        # Account for centering
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :] = proj.T @ mat - centering

    return projections, projected_X


def parafac2_nd(
    X: anndata.AnnData,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-7,
    verbose=False,
    random_state=None,
) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], float]:
    r"""The same interface as regular PARAFAC2."""
    # Index dataset to a list of conditions
    sgIndex = X.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cond = np.amax(sgIndex) + 1

    Xarr = sps.csr_array(X.X)
    means = cp.array(X.var["means"].to_numpy())

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed

    # Calculate the norm of the dataset
    norm_tensor = calc_total_norm(X)

    xInit = cupy_sparse.csr_matrix(Xarr[::3])

    cp.random.set_random_state(cp.random.RandomState(random_state))
    _, _, C = svds(xInit, k=rank, return_singular_vectors=True)

    tl.set_backend("cupy")

    factors = [
        cp.ones((n_cond, rank)),
        cp.eye(rank),
        C.T,
    ]

    errs: list[float] = []
    err = float("NaN")

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        lineIter = iter % 2 == 0 and iter > 5

        # Initiate line search
        if lineIter:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            factors_ls = [
                factors_old[ii] + (factors[ii] - factors_old[ii]) * jump  # type: ignore
                for ii in range(3)
            ]

            projections_ls, projected_X_ls = project_data(Xarr, sgIndex, means, factors)
            err_ls = reconstruction_error(
                factors_ls, projections_ls, projected_X_ls, norm_tensor
            )

            if err_ls < errs[-1] * norm_tensor:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                factors = factors_ls
            else:
                lineIter = False
                acc_fail += 1

                if acc_fail >= 4:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        if lineIter is False:
            projections, projected_X = project_data(Xarr, sgIndex, means, factors)
            err = reconstruction_error(factors, projections, projected_X, norm_tensor)

        errs.append(err / norm_tensor)

        factors_old = deepcopy(factors)
        _, factors = parafac(
            projected_X,  # type: ignore
            rank,
            n_iter_max=5,
            tol=False,
            normalize_factors=False,
            l2_reg=0.01,  # type: ignore
            init=(None, factors),  # type: ignore
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    tl.set_backend("numpy")

    factors = [f.get() for f in factors]
    projections = [p.get() for p in projections]  # type: ignore

    print(f"Degeneracy score: {degeneracy_score((None, factors))}")

    return standardize_pf2(factors, projections), 1.0 - errs[-1]


def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections
