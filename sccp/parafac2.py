from warnings import warn

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition._cp import initialize_cp
from tensorly.parafac2_tensor import parafac2_to_slice, _validate_parafac2_tensor
from tensorly.cp_tensor import cp_normalize
from tensorly.decomposition import non_negative_parafac_hals


def _project_tensor_slices_fused(tensor_slices, projections):
    return np.einsum("ijk,ijl->ilk", tensor_slices, projections)


def to_three_mode(X, factors):
    """Take an N-mode PARAFAC2 model, and reshape to 3-mode."""
    X_r = np.reshape(X.T, (-1, X.shape[-2], X.shape[-1]))
    new_factors = [khatri_rao(factors[:-2]), factors[-2], factors[-1]]
    return X_r, new_factors


def _compute_projections_fused(X, factors):
    svd_i = np.einsum("ij,lj,kj,iak->ila", *factors, X)

    n_eigenvecs = factors[0].shape[1]
    min_dim = min(svd_i.shape[1], svd_i.shape[2])
    U, _, Vh = np.linalg.svd(svd_i, full_matrices=n_eigenvecs > min_dim)
    U = U[:, :, :n_eigenvecs]
    Vh = Vh[:, :n_eigenvecs, :]

    return np.einsum("ijk,ikl->ilj", U, Vh)


def _compute_projections(X, factors):
    X_r, new_factors = to_three_mode(X, factors)
    projections = _compute_projections_fused(X_r, new_factors)
    p_t = _project_tensor_slices_fused(X_r, projections)

    proj_tensor_reshape = np.reshape(p_t, (*X.shape[:-2], p_t.shape[-2], p_t.shape[-1]))
    return projections, proj_tensor_reshape


def _parafac2_rec_error(X, decomposition):
    X_r, new_factors = to_three_mode(X, decomposition[1])
    new_decomp = (decomposition[0], new_factors, decomposition[2])
    _validate_parafac2_tensor(new_decomp)

    squared_error = 0
    for idx, tensor_slice in enumerate(X_r):
        reconstruction = parafac2_to_slice(new_decomp, idx, validate=False)
        squared_error += tl.sum((tensor_slice - reconstruction) ** 2)
    return squared_error


def parafac2(
    X_slices,
    rank,
    n_iter_max=150,
    tol=1e-8,
    nn_modes=None,
    verbose=False,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    _, factors = initialize_cp(X_slices, rank, init="random", normalize_factors=False)
    factors[-2] = np.eye(rank)
    projections = _compute_projections(X_slices, factors)

    errs = []
    norm_tensor = np.linalg.norm(X_slices) ** 2

    for iter in range(n_iter_max):
        projections, projected_tensor = _compute_projections(X_slices, factors)
        _, factors = non_negative_parafac_hals(
            projected_tensor,
            rank,
            n_iter_max=50,
            init=(None, factors),
            nn_modes=nn_modes,
            verbose=False,
            return_errors=False,
            tol=1e-100,
        )

        rec_error = _parafac2_rec_error(X_slices, (None, factors, projections))
        errs.append(rec_error / norm_tensor)

        if iter >= 1:
            if verbose:
                print(f"iteration {iter}: error={errs[-1]}, Δ={errs[-2] - errs[-1]}.")

            if (errs[-2] - errs[-1]) < (tol * errs[-2]) or errs[-1] < 1e-12:
                if verbose:
                    print(f"converged in {iter} iterations.")
                break
        else:
            if verbose:
                print(f"iteration 1: error={errs[-1]}.")

    weights, factors = cp_normalize((None, factors))
    return weights, factors, projections
