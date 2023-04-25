from copy import deepcopy
import numpy as np
from tensorly.tenalg.svd import randomized_svd
from .parafac2 import parafac2_nd, _cmf_reconstruction_error
from tensorly.parafac2_tensor import parafac2_to_slices


def crossvalidate_PCA(X: np.ndarray, rank: int, trainPerc: float = 0.75) -> np.ndarray:
    """Bi-cross-validation for PCA. Because PCA is component-by-component, this
    provides R2X for all ranks in one pass."""
    # Shuffle so that we take a different subset each time
    X = X.copy()
    X = X[np.random.permutation(X.shape[0]), :]
    X = X[:, np.random.permutation(X.shape[1])]

    # Indices where we will split
    X_B_idx = int(X.shape[0] * trainPerc)
    X_C_idx = int(X.shape[1] * trainPerc)

    # All genes, not all cells
    Xb = X[:X_B_idx, :]
    # All cells, not all genes
    Xc = X[:, :X_C_idx]

    mean_ = np.mean(Xb, axis=0)
    loadings = randomized_svd(Xb - mean_, rank)[2]
    scores = (Xc - mean_[:X_C_idx]) @ loadings[:, :X_C_idx].T

    total_var = float(np.linalg.norm(X[X_B_idx:, X_C_idx:]) ** 2)

    # Reconstruct and get error
    recon_error = np.zeros(rank)

    for i in range(rank):
        X_err = X - (scores[:, :(i + 1)] @ loadings[:(i + 1), :] + mean_)
        recon_error[i] = float(np.linalg.norm(X_err[X_B_idx:, X_C_idx:]) ** 2)

    return 1.0 - recon_error / total_var


def crossvalidate(X, rank: int, trainPerc: float = 0.75, verbose: bool = True) -> float:
    # Shuffle, rnd.shuffle handles the cell axis
    var_idx = np.random.permutation(X[0].shape[1])
    X = [xx[:, var_idx] for xx in X]
    for xx in X:
        np.random.shuffle(xx)

    X_B_idx = [int(xx.shape[0] * trainPerc) for xx in X]
    B_train = [X[ii][:bi, :] for ii, bi in enumerate(X_B_idx)]

    X_C_idx = int(X[0].shape[1] * trainPerc)
    C_train = [xx[:, :X_C_idx] for xx in X]

    w_B, fac_B, _, _ = parafac2_nd(B_train, rank, verbose=verbose)

    fac_C = deepcopy(fac_B)
    fac_C[0] *= w_B[np.newaxis, :]
    fac_C[2] = fac_C[2][:X_C_idx, :]
    _, proj = _cmf_reconstruction_error(C_train, fac_C, 1.0)

    # Project projections into B space
    X_recon = parafac2_to_slices((w_B, fac_B, proj), validate=False)

    recon_error = 0.0
    total_var = 0.0
    for ii in range(len(X_recon)):
        xr = X_recon[ii][X_B_idx[ii] :, X_C_idx:]
        xx = X[ii][X_B_idx[ii] :, X_C_idx:]

        recon_error += float(np.linalg.norm(xx - xr) ** 2)
        total_var += float(np.linalg.norm(xx) ** 2)

    return 1.0 - recon_error / total_var


def CrossVal(X, rank: int, trainPerc: float = 0.75):
    """Creates cross validation accuracy plot for parafac2"""
    rank_vec = np.arange(1, rank + 1)

    # Collect Pf2 results
    cv_pf2_error = [crossvalidate(X.X_list, rank=rank, trainPerc=trainPerc) for r in rank_vec]
    cv_pca_error = crossvalidate_PCA(X.unfold(), rank, trainPerc = trainPerc)

    return cv_pf2_error, cv_pca_error
