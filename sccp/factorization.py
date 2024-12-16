import anndata
import numpy as np
import scipy.sparse as sps
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2
from scipy.stats import gmean
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]
    counts = np.zeros((np.amax(sgIndex) + 1, 1))

    cond_mean = gmean(X.uns["Pf2_A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct


def pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
    tolerance=1e-9,
):
    """Run Pf2 model and store results in anndata file"""
    pf_out, _ = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=500
    )

    X = store_pf2(X, pf_out)

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def pf2_pca_r2x(X: anndata.AnnData, ranks):
    """Run Pf2/PCA on data and save R2X values"""
    X = X.to_memory()
    XX = sps.csr_array(X.X)

    r2x_pf2 = np.zeros(len(ranks))

    for i in tqdm(range(len(r2x_pf2)), total=len(r2x_pf2)):
        _, R2X = parafac2_nd(X, rank=i + 1)
        r2x_pf2[i] = R2X

    pca = PCA(n_components=ranks[-1], svd_solver="arpack")
    pca.fit(XX)
    r2x_pca = np.cumsum(pca.explained_variance_ratio_)
    r2x_pca = r2x_pca[np.array(ranks) - 1]
    
    
    for i, rank in enumerate(ranks):
        print(f"Rank {rank}: R2X (PF2) = {r2x_pf2[i]:.4f}, R2X (PCA) = {r2x_pca[i]:.4f}")
    
    for i, rank in enumerate(ranks):
        assert r2x_pf2[i] < r2x_pca[i], f"R2X for rank {rank} is not lower than PCA"

    return r2x_pf2, r2x_pca
