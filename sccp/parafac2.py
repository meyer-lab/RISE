from warnings import warn

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import cp_normalize
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.decomposition._parafac2 import _compute_projections as tl_proj, _project_tensor_slices


def _compute_projections(X, factors):
    X_r = np.reshape(X, (X.shape[0], X.shape[1], -1))

    new_factors = [factors[0], factors[1], khatri_rao(factors[2:])]
    projections = tl_proj(X_r, new_factors, tl.partial_svd)
    p_t = _project_tensor_slices(X_r, projections)

    proj_tensor_reshape = np.reshape(p_t, (p_t.shape[0], p_t.shape[1], *X.shape[2:]))
    return projections, proj_tensor_reshape


def parafac2(
    tensor_slices,
    rank,
    n_iter_max=2000,
    tol=1e-8,
    nn_modes=None,
    verbose=False,
    n_iter_parafac=5,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    weights, factors = initialize_cp(tensor_slices, rank)
    factors[1] = np.eye(rank)
    projections = _compute_projections(tensor_slices, factors)

    errs = []
    norm_tensor = tl.sqrt(
        sum(tl.norm(tensor_slice, 2) ** 2 for tensor_slice in tensor_slices)
    )

    # If nn_modes is set, we use HALS, otherwise, we use the standard parafac implementation.
    if nn_modes is not None and (nn_modes == "all" or 1 in nn_modes):
        warn(
            "Mode `1` of PARAFAC2 fitted with ALS cannot be constrained to be truly non-negative. See the documentation for more info."
        )

    def parafac_updates(X, w, f):
        return non_negative_parafac_hals(
            X,
            rank,
            n_iter_max=n_iter_parafac,
            init=(w, f),
            nn_modes=nn_modes,
            verbose=verbose,
            return_errors=True,
            tol=1e-100,
        )

    for iter in range(n_iter_max):
        factors[1] = factors[1] * np.reshape(weights, (1, -1))
        weights = np.ones(weights.shape)

        projections, projected_tensor = _compute_projections(tensor_slices, factors)
        (_, factors), cp_error = parafac_updates(projected_tensor, weights, factors)

        weights, factors = cp_normalize((weights, factors))

        errs.append(cp_error[-1] / norm_tensor)

        if iter >= 1:
            if verbose:
                print(
                    f"iteration {iter}: reconstruction error={errs[-1]}, diff={errs[-2] - errs[-1]}."
                )

            if abs(errs[-2] - errs[-1]) < (tol * errs[-2]) or errs[-1] < 1e-12:
                if verbose:
                    print(f"converged in {iter} iterations.")
                break
        else:
            if verbose:
                print(f"iteration 1: reconstruction error={errs[-1]}.")

    return weights, factors, projections
