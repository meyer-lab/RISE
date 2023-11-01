import os
from tqdm import tqdm
from copy import deepcopy
from typing import Sequence
import numpy as np
import anndata
import tensorly as tl
import torch
from pacmap import PaCMAP
from .imports import import_citeseq, import_lupus, import_thomson
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.cp_tensor import cp_flip_sign, CPTensor, cp_normalize
from tensorly.tenalg.svd import randomized_svd, truncated_svd
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error, _compute_projections, _project_tensor_slices
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

    weight, factors, projs, r2x = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.uns["cov_ratio"] = cov_var / cov_total
    X.uns["cvSNR"] = cwSNR(X_pf, weight, factors, projs)
    X.uns["R2X"] = r2x

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


def _cmf_reconstruction_error(matrices: Sequence, factors: list, norm_X_sq):
    A, B, C = factors

    norm_sq_err = norm_X_sq
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        if isinstance(B, torch.Tensor):
            mat_gpu = torch.tensor(mat).cuda().double()
        else:
            mat_gpu = mat

        lhs = B @ (A[i] * C).T
        U, _, Vh = truncated_svd(mat_gpu @ lhs.T, A.shape[1])
        proj = U @ Vh

        projections.append(proj)
        projected_X.append(proj.T @ mat_gpu)

        B_i = (proj @ B) * A[i]

        # trace of the multiplication products
        norm_sq_err -= 2.0 * tl.trace(A[i][:, np.newaxis] * B.T @ projected_X[-1] @ C)
        norm_sq_err += ((B_i.T @ B_i) * CtC).sum()

    return norm_sq_err, projections, projected_X


@torch.inference_mode()
def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-6,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed

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

    tl.set_backend("pytorch")
    CP = CPTensor(
        (
            None,
            [
                tl.ones((len(X_in), rank)).cuda().double(),
                tl.eye(rank).cuda().double(),
                torch.tensor(C).cuda().double(),
            ],
        )
    )

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X_in, CP.factors, norm_tensor
        )

        # Initiate line search
        if iter % 2 == 0 and iter > 5:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            CP_ls = deepcopy(CP)
            CP_ls.factors = [
                CP_old.factors[ii] + (CP.factors[ii] - CP_old.factors[ii]) * jump
                for ii in range(3)
            ]
            err_ls, projections_ls, projected_X_ls = _cmf_reconstruction_error(
                X_in, CP_ls.factors, norm_tensor
            )

            if err_ls < err:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                CP = CP_ls
            else:
                acc_fail += 1

                if acc_fail >= 4:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        errs.append(tl.to_numpy((err / norm_tensor).cpu()))

        # Project tensor slices
        projected_X = tl.stack(projected_X)

        CP_old: CPTensor = deepcopy(CP)
        CP = non_negative_parafac_hals(
            projected_X,
            rank,
            n_iter_max=10,
            nn_modes=(0, ),
            init=CP,
            tol=False,
            normalize_factors=False,
        )

        if iter > 1:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    gini_idx = giniIndex(tl.to_numpy(CP.factors[0].cpu()))
    assert gini_idx.size == rank

    CP.factors = [f.numpy(force=True)[:, gini_idx] for f in CP.factors]
    CP.weights = CP.weights.numpy(force=True)[gini_idx]

    CP = cp_normalize(cp_flip_sign(CP, mode=1))

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(CP.factors[1].T), maximize=True)
    CP.factors[1] = CP.factors[1][col_ind, :]
    projections = [p.numpy(force=True)[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(CP.factors[1]))
    CP.factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return CP.weights, CP.factors, projections, R2X


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment"""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)
