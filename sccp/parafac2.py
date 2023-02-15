import numpy as np
from tensorly.tenalg import khatri_rao
from opt_einsum import contract
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.decomposition._parafac2 import (
    _project_tensor_slices,
    _compute_projections,
    initialize_decomposition,
)
import tlviz


def _parafac2_reconstruction_error(X, decomposition):
    (A, B, C), projections = decomposition
    proj = np.stack(projections, axis=0)

    reconstruction = contract("ijm,mk,ik,lk->ijl", proj, B, A, C)
    return np.linalg.norm(X - reconstruction)


def parafac2_nd(
    X_nd,
    rank,
    n_iter_max=100,
    init="svd",
    svd="randomized_svd",
    tol=1e-6,
    nn_modes=None,
    random_state=None,
    verbose=False,
    n_iter_parafac=30,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    # Unless otherwise labelled, variables are structured as 3D.
    X = np.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))
    _, factors, projections = initialize_decomposition(
        X, rank, init=init, svd=svd, random_state=random_state
    )

    errs = []
    norm_tensor = np.linalg.norm(X) ** 2

    for iter in range(n_iter_max):
        projections = _compute_projections(X, factors, svd)
        projected_tensor = _project_tensor_slices(X, projections)

        # Convert projections and projected tensor to nD
        projections_nD = np.reshape(projections, (*X_nd.shape[0:-1], rank))
        projected_tensor_nD = np.reshape(
            projected_tensor, (*X_nd.shape[0:-2], rank, X_nd.shape[-1])
        )

        _, factors_nD = non_negative_parafac_hals(
            projected_tensor_nD,
            rank,
            n_iter_max=n_iter_parafac,
            init=init if iter == 0 else (None, factors_nD),
            svd=svd,
            nn_modes=nn_modes,
            tol=False,
        )

        # Convert factors to 3D
        factors = [khatri_rao(factors_nD[:-2]), factors_nD[-2], factors_nD[-1]]

        rec_error = _parafac2_reconstruction_error(X, (factors, projections))

        errs.append(rec_error / norm_tensor)

        if iter >= 1:
            if verbose:
                print(
                    f"iteration {iter + 1}: error={errs[-1]}, Δ={errs[-2] - errs[-1]}."
                )

            if (errs[-2] - errs[-1]) < (tol * errs[-2]) or errs[-1] < 1e-12:
                if verbose:
                    print(f"converged in {iter + 1} iterations.")
                break
        else:
            if verbose:
                print(f"iteration 1: error={errs[-1]}.")

    r2x = 1 - np.square(_parafac2_reconstruction_error(
            X, (factors, projections))) / np.square(np.linalg.norm(X))
    
    weights, factors_nD = cp_normalize((None, factors_nD))
    weights, factors_nD = cp_flip_sign((weights, factors_nD), mode=1)

    coreC = tlviz.model_evaluation.core_consistency((weights, factors_nD), projected_tensor_nD, normalised=True)
    print(f"Core consistency = {coreC}.")

    return weights, factors_nD, projections_nD, r2x, coreC
