import numpy as np
from sklearn.decomposition import PCA
from .parafac2 import parafac2_nd
from tensorly.parafac2_tensor import parafac2_to_slices


def permute_sign_proc(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Apply the procrustes algorithm to find the best match
    S, _, Vh = np.linalg.svd(X.T @ Y)
    procM = (S @ Vh).T

    # Limit to permutation and sign flips
    procM[np.abs(procM) < 0.5] = 0.0
    procM = np.sign(procM)
    return procM


def crossvalidate_PCA(X: np.ndarray, rank: int, trainPerc: float=0.75):
    X = X.copy()
    pc = PCA(n_components=rank)

    # Shuffle so that we take a different subset each time
    X = X[np.random.permutation(np.arange(X.shape[0])), :]
    X = X[:, np.random.permutation(np.arange(X.shape[1]))]

    # Indices where we will split
    X_B_idx = int(X.shape[0] * trainPerc)
    X_C_idx = int(X.shape[1] * trainPerc)

    # All cells, not all genes
    C_scores = pc.fit_transform(X[:, :X_C_idx])
    # All genes, not all cells
    B_scores = pc.fit_transform(X[:X_B_idx, :])

    procM = permute_sign_proc(B_scores, C_scores[:X_B_idx, :])
    X_recon = pc.inverse_transform(C_scores @ procM)

    # Reconstruct and get error
    X_err = X - X_recon

    recon_error = np.linalg.norm(X_err[X_B_idx:, X_C_idx:]) ** 2
    total_var = np.linalg.norm(X[X_B_idx:, X_C_idx:]) ** 2

    return 1.0 - recon_error / total_var


def crossvalidate(X, rank: int, trainPerc: float=0.75, verbose=True):
    # Shuffle, rnd.shuffle handles the cell axis
    var_idx = np.random.permutation(np.arange(X[0].shape[1]))
    X = [xx[:, var_idx] for xx in X]
    for xx in X:
        np.random.shuffle(xx)

    X_B_idx = [int(xx.shape[0] * trainPerc) for xx in X]
    B_train = [X[ii][:bi, :] for ii, bi in enumerate(X_B_idx)]

    X_C_idx = int(X[0].shape[1] * trainPerc)
    C_train = [xx[:, :X_C_idx] for xx in X]

    w_B, fac_B, proj_B, _ = parafac2_nd(B_train, rank, verbose=verbose)
    _, _, proj_C, _ = parafac2_nd(C_train, rank, verbose=verbose)

    # Solve procrustes to project C onto B
    proj_B_flat = np.concatenate(proj_B, axis=0)
    proj_C_flat = np.concatenate([proj_C[ii][:bi, :] for ii, bi in enumerate(X_B_idx)], axis=0)
    procM = permute_sign_proc(proj_B_flat, proj_C_flat)

    # Project projections into B space
    X_recon = parafac2_to_slices((w_B, fac_B, [cc @ procM for cc in proj_C]), validate=False)

    recon_error = 0.0
    total_var = 0.0
    for ii in range(len(X_recon)):
        xr = X_recon[ii][X_B_idx[ii]:, X_C_idx:]
        xx = X[ii][X_B_idx[ii]:, X_C_idx:]

        recon_error += np.linalg.norm(xx - xr) ** 2
        total_var += np.linalg.norm(xx) ** 2

    return 1.0 - recon_error / total_var


def plotCrossVal(X, rank, ax1, trainPerc=0.75):
    """Creates cross validation accuracy plot for parafac2"""
    rank_vec = np.arange(1, rank + 1)

    # Collect Pf2 results
    cv_error = [crossvalidate(X, rank=r, trainPerc=trainPerc) for r in rank_vec]

    ax1.scatter(rank_vec, cv_error, marker="x", s=20.0)

    ax1.set(
        ylabel="CV Accuracy",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-0.05, np.max(cv_error) + 0.05),
    )
