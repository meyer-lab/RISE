import numpy as np
from sklearn.decomposition import PCA
from .parafac2 import parafac2_nd
from tensorly.parafac2_tensor import parafac2_to_slices


def parafac2_to_tensor(parafac2_tensor):
    _, (A, _, C), projections = parafac2_tensor
    slices = parafac2_to_slices(parafac2_tensor, validate=False)
    lengths = [projection.shape[0] for projection in projections]

    tensor = np.zeros((A.shape[0], max(lengths), C.shape[0]))
    for i, (slice_, length) in enumerate(zip(slices, lengths)):
        tensor[i, :length] = slice_
    return tensor


def shuffle_X(X):
    X = np.copy(X)

    A_idx = np.random.permutation(np.arange(X.shape[0]))
    X = X[A_idx, :, :]

    C_idx = np.random.permutation(np.arange(X.shape[2]))
    X = X[:, :, C_idx]

    for i in range(X.shape[0]):
        B_idx = np.random.permutation(np.arange(X.shape[1]))
        X[i, :, :] = X[i, B_idx, :]

    return X


def permute_sign_proc(X, Y):
    # Apply the procrustes algorithm to find the best match
    S, _, Vh = np.linalg.svd(X.T @ Y)
    procM = (S @ Vh).T

    # Limit to permutation and sign flips
    procM[np.abs(procM) < 0.5] = 0.0
    procM = np.sign(procM)
    return procM


def crossvalidate_PCA(X, rank, trainPerc=0.75):
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


def crossvalidate(X, rank, trainPerc=0.75, verbose=True):
    X = shuffle_X(X)

    X_B_idx = int(X.shape[1] * trainPerc)
    B_train = X[:, :X_B_idx, :]

    X_C_idx = int(X.shape[2] * trainPerc)
    C_train = X[:, :, :X_C_idx]

    w_B, fac_B, proj_B, _ = parafac2_nd(B_train, rank, verbose=verbose)
    _, _, proj_C, _ = parafac2_nd(C_train, rank, verbose=verbose)

    # Solve procrustes to project C onto B
    proj_B_flat = np.reshape(proj_B, (-1, proj_B.shape[2]))
    proj_C_flat = np.reshape(proj_C[:, :X_B_idx, :], (-1, proj_C.shape[2]))
    procM = permute_sign_proc(proj_B_flat, proj_C_flat)

    # Project projections into B space
    X_reconst = parafac2_to_tensor((w_B, fac_B, proj_C @ procM))
    X_reconst = np.stack(X_reconst, axis=0)
    X_err = X_reconst - X

    recon_error = np.linalg.norm(X_err[:, X_B_idx:, X_C_idx:]) ** 2
    total_var = np.linalg.norm(X[:, X_B_idx:, X_C_idx:]) ** 2

    return 1.0 - recon_error / total_var


def plotCrossVal(tensor, rank, ax1, trainPerc=0.75):
    """Creates cross validation accuracy plot for parafac2"""
    rank_vec = np.arange(1, rank + 1)
    cv_error = np.empty(len(rank_vec))

    # Collect Pf2 results
    for i in range(len(rank_vec)):
        cv_error[i] = crossvalidate(tensor, rank=i + 1, trainPerc=trainPerc)

    ax1.scatter(rank_vec, cv_error, marker="x", s=20.0)

    ax1.set(
        ylabel="CV Accuracy",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-0.05, np.max(cv_error) + 0.05),
    )
