import numpy as np
from tensorly.tenalg import khatri_rao
from tensorly.cp_tensor import cp_normalize, cp_flip_sign
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.decomposition._parafac2 import (
    _project_tensor_slices,
    _compute_projections,
    _parafac2_reconstruction_error,
    parafac2,
)


def parafac2_nd(
    X_nd,
    rank,
    n_iter_max=2000,
    init="svd",
    svd="truncated_svd",
    tol=1e-8,
    nn_modes=None,
    random_state=None,
    verbose=False,
    n_iter_parafac=20,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    # Unless otherwise labelled, variables are structured as 3D.
    if verbose:
        print("We are going to start by just running 3D PARAFAC2 for 1 iteration.")

    X = np.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))

    weights, factors, projections = parafac2(
        X,
        rank,
        n_iter_max=1,
        init=init,
        svd=svd,
        normalize_factors=True,
        tol=tol,
        absolute_tol=1e-13,
        nn_modes=nn_modes,
        random_state=random_state,
        verbose=False,
        n_iter_parafac=n_iter_parafac,
    )

    errs = []
    norm_tensor = np.linalg.norm(X) ** 2

    for iter in range(n_iter_max):
        factors[1] = factors[1] * np.reshape(weights, (1, -1))
        weights = np.ones(weights.shape)

        projections = _compute_projections(X, factors, svd)
        projected_tensor = _project_tensor_slices(X, projections)

        # Convert projections and projected tensor to nD
        projections_nD = np.reshape(projections, (*X_nd.shape[0:-1], rank))
        projected_tensor_nD = np.reshape(
            projected_tensor, (*X_nd.shape[0:-2], rank, X_nd.shape[-1])
        )

        factors_nD = non_negative_parafac_hals(
            projected_tensor_nD,
            rank,
            n_iter_max=n_iter_parafac,
            init="svd" if iter == 0 else (weights, factors_nD),
            svd=svd,
            nn_modes=nn_modes,
            verbose=False,
            return_errors=False,
            tol=1e-100,
        )[1]
        weights, factors_nD = cp_normalize((None, factors_nD))
        weights, factors_nD = cp_flip_sign((weights, factors_nD), mode=1)

        # Convert factors to 3D
        factors = [khatri_rao(factors_nD[:-2]), factors_nD[-2], factors_nD[-1]]

        rec_error = _parafac2_reconstruction_error(X, (weights, factors, projections))

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

    return weights, factors_nD, projections_nD
