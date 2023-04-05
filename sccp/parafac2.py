import torch
import numpy as np
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import randomized_svd
from tensorly.decomposition import parafac
from tensorly.decomposition._parafac2 import _compute_projections


class Pf2X:
    def __init__(self, X_list, condition_labels, variable_labels):
        assert isinstance(X_list, list)
        self.X_list = X_list
        self.condition_labels = np.array(condition_labels, dtype=object)
        self.variable_labels = np.array(variable_labels, dtype=object)
        assert len(X_list) == len(condition_labels)
        for X in X_list:
            assert X.shape[1] == len(variable_labels)

    def unfold(self):
        return tl.concatenate(self.X_list, axis=0)


def _cmf_reconstruction_error(matrices, decomposition, norm_X_sq):
    (A, B, C), projections = decomposition

    norm_cmf_sq = 0
    inner_product = 0
    CtC = C.T @ C

    for i, proj in enumerate(projections):
        B_i = (proj @ B) * A[i]
        # trace of the multiplication products
        inner_product += tl.einsum("ji,jk,ki", B_i, matrices[i], C)
        norm_cmf_sq += tl.sum((B_i.T @ B_i) * CtC)

    return norm_X_sq - 2 * inner_product + norm_cmf_sq


@torch.inference_mode()
def parafac2_nd(
    X,
    rank: int,
    n_iter_max: int = 200,
    tol: float=1e-9,
    verbose: bool=False,
):
    r"""The same interface as regular PARAFAC2."""
    tl.set_backend("pytorch")
    if isinstance(X, Pf2X):
        X = X.X_list

    norm_tensor = np.sum([np.linalg.norm(xx) ** 2 for xx in X])
    X = [tl.tensor(xx).cuda() for xx in X]

    # Initialization
    unfolded = tl.concatenate(list(X), axis=0).T
    assert tl.shape(unfolded)[0] > rank
    C = randomized_svd(unfolded, rank)[0]
    CP = tl.cp_tensor.CPTensor((None, [tl.ones((len(X), rank)).cuda(), tl.eye(rank).cuda(), C]))
    projections = _compute_projections(X, CP.factors, "truncated_svd")

    errs = []

    err = _cmf_reconstruction_error(X, (CP.factors, projections), norm_tensor)
    errs.append(tl.to_numpy(err.cpu()) / norm_tensor)

    tq = tqdm(range(n_iter_max), disable=(not verbose), delay=1, mininterval=1)
    for _ in tq:
        # Push the genes factors to be orthogonal
        CP.factors[2] = tl.qr(CP.factors[2])[0]

        projections = _compute_projections(X, CP.factors, "truncated_svd")

        # Project tensor slices
        projected_X = tl.stack([p.T @ t for t, p in zip(X, projections)])

        CP = parafac(
            projected_X,
            rank,
            n_iter_max=5,
            init=CP,
            tol=1e-100,
            normalize_factors=False,
        )

        err = _cmf_reconstruction_error(X, (CP[1], projections), norm_tensor)
        errs.append(tl.to_numpy(err.cpu()) / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

        if delta < tol:
            break

    CP = cp_normalize(CP)
    CP = cp_flip_sign(CP, mode=1)

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    weights = tl.to_numpy(CP[0].cpu())
    factors = [tl.to_numpy(f.cpu()) for f in CP[1]]
    projections = [tl.to_numpy(p.cpu()) for p in projections]
    return weights, factors, projections, R2X
