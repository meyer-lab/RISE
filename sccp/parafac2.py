import torch
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import parafac


def _compute_projections(tensor_slices, factors):
    n_eig = factors[0].shape[1]
    out = []

    for A, tensor_slice in zip(factors[0], tensor_slices):
        lhs = factors[1] @ (A * factors[2]).T
        rhs = tensor_slice.T
        U, _, Vh = tl.truncated_svd(lhs @ rhs, n_eigenvecs=n_eig)

        out.append((U @ Vh).T)

    return out


def _cmf_reconstruction_error(matrices, decomposition, norm_X_sq):
    (A, B, C), projections = decomposition

    norm_cmf_sq = 0
    inner_product = 0
    CtC = C.T @ C

    for i, proj in enumerate(projections):
        B_i = tl.dot(proj, B) * A[i]
        # trace of the multiplication products
        inner_product += tl.einsum("ji,jk,ki", B_i, matrices[i], C)
        norm_cmf_sq += tl.sum((B_i.T @ B_i) * CtC)

    return norm_X_sq - 2 * inner_product + norm_cmf_sq


@torch.inference_mode()
def parafac2_nd(
    X_nd,
    rank: int,
    n_iter_max: int=200,
    tol=1e-9,
    verbose=False,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    tl.set_backend("pytorch")
    X_nd = tl.tensor(X_nd).cuda()
    X = tl.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))

    # Initialization
    unfolded = tl.unfold(X, 2)
    assert tl.shape(unfolded)[0] > rank
    C = torch.svd_lowrank(unfolded, rank)[0]
    factors = [tl.ones((X.shape[0], rank)).cuda(), tl.eye(rank).cuda(), C]
    projections = _compute_projections(X, factors)

    errs = []
    norm_tensor = tl.norm(X) ** 2

    err = _cmf_reconstruction_error(X, (factors, projections), norm_tensor)
    errs.append(tl.to_numpy((err / norm_tensor).cpu()))

    CP_nD = "svd"

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for _ in tq:
        projections = _compute_projections(X, factors)
        projections = tl.stack(projections, axis=0)
        projections_nD = tl.reshape(projections, (*X_nd.shape[0:-1], rank))

        # Project tensor slices
        projected_X_nD = tl.einsum("...ji,...jk->...ik", projections_nD, X_nd)

        CP_nD = parafac(
            projected_X_nD,
            rank,
            n_iter_max=10,
            init=CP_nD,
            tol=1e-100,
            normalize_factors=False,
        )

        # Convert factors to 3D
        factors = [khatri_rao(CP_nD.factors[:-2]), CP_nD.factors[-2], CP_nD.factors[-1]]

        err = _cmf_reconstruction_error(X, (factors, projections), norm_tensor)
        errs.append(tl.to_numpy((err / norm_tensor).cpu()))

        delta = errs[-2] - errs[-1]
        tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

        if delta < tol:
            break

    CP_nD = cp_normalize(CP_nD)
    CP_nD = cp_flip_sign(CP_nD, mode=1)

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    weights = tl.to_numpy(CP_nD[0].cpu())
    factors_nD = [tl.to_numpy(f.cpu()) for f in CP_nD[1]]
    projections_nD = tl.to_numpy(projections_nD.cpu())
    return weights, factors_nD, projections_nD, R2X
