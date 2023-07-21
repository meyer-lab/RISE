from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg.svd import randomized_svd
from tensorly.decomposition import parafac
from scipy.optimize import linear_sum_assignment


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


def _cmf_reconstruction_error(matrices, factors: list, norm_X_sq, rng=None):
    A, B, C = factors

    norm_cmf_sq = 0
    inner_product = 0
    CtC = C.T @ C
    projections = []
    projected_X = []

    for i, mat in enumerate(matrices):
        if isinstance(B, torch.Tensor):
            mat_gpu = torch.tensor(mat).cuda().double()
        else:
            mat_gpu = mat

        lhs = B @ (A[i] * C).T
        U, _, Vh = randomized_svd(mat_gpu @ lhs.T, A.shape[1], random_state=rng)
        proj = U @ Vh

        B_i = (proj @ B) * A[i]
        # trace of the multiplication products
        inner_product += tl.trace(B_i.T @ mat_gpu @ C)
        norm_cmf_sq += tl.sum((B_i.T @ B_i) * CtC)
        projections.append(proj)
        projected_X.append(proj.T @ mat_gpu)

    return norm_X_sq - 2 * inner_product + norm_cmf_sq, projections, projected_X


@torch.inference_mode()
def parafac2_nd(
    X_in,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
    random_state=None,
    linesearch=True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed
    max_fail: int = 4  # Increase acc_pow with one after max_fail failure

    if isinstance(X_in, Pf2X):
        X_in = X_in.X_list

    X = X_in

    norm_tensor = np.sum([np.linalg.norm(xx) ** 2 for xx in X_in])

    # Checks size of each experiment is bigger than rank
    for i in range(len(X)):
        assert np.shape(X[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X[0])[1] > rank

    # Initialization  
    unfolded = np.concatenate(list(X), axis=0).T
    C = randomized_svd(unfolded, rank, random_state=rng)[0]

    tl.set_backend("pytorch")
    CP = tl.cp_tensor.CPTensor(
        (
            None,
            [tl.ones((len(X), rank)).cuda().double(), tl.eye(rank).cuda().double(), torch.tensor(C).cuda().double()],
        )
    )

    errs = []

    tq = tqdm(range(n_iter_max), disable=(not verbose), mininterval=2)
    for iter in tq:
        err, projections, projected_X = _cmf_reconstruction_error(
            X, CP.factors, norm_tensor, rng=rng
        )

        # Will we be performing a line search iteration
        if linesearch and iter % 2 == 0 and iter > 5:
            line_iter = True
        else:
            line_iter = False


        # Initiate line search
        if line_iter:
            jump = iter ** (1.0 / acc_pow)

            # Estimate error with line search
            CP_ls = deepcopy(CP)
            CP_ls.factors = [
                CP_old.factors[ii] + (CP.factors[ii] - CP_old.factors[ii]) * jump
                for ii in range(3)
            ]
            err_ls, projections_ls, projected_X_ls = _cmf_reconstruction_error(
                X, CP_ls.factors, norm_tensor, rng=rng
            )

            if err_ls < err:
                acc_fail = 0
                err = err_ls
                projections = projections_ls
                projected_X = projected_X_ls
                CP = CP_ls

                if verbose:
                    print(f"Accepted line search jump of {jump}.")
            else:
                acc_fail += 1

                if verbose:
                    print(f"Line search failed for jump of {jump}.")

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        errs.append(tl.to_numpy((err / norm_tensor).cpu()))

        # Project tensor slices
        projected_X = tl.stack(projected_X)

        CP_old: CPTensor = deepcopy(CP)
        CP = parafac(
            projected_X,
            rank,
            n_iter_max=2,
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

    for ii in range(3):
        np.testing.assert_allclose(
            np.linalg.norm(CP.factors[ii], axis=0), 1.0, rtol=1e-2
        )

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
