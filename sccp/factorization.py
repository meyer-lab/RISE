from pacmap import PaCMAP
from sklearn.linear_model import LinearRegression
from scipy.stats import gmean
from tlviz.factor_tools import degeneracy_score
from parafac2.parafac2 import parafac2_nd
from sklearn.decomposition import PCA
import anndata
import scipy.sparse as sps
import numpy as np
from tqdm import tqdm


def cw_snr(
    X: anndata.AnnData,
) -> np.ndarray:
    """Calculate the columnwise signal-to-noise ratio for each dataset and overall error."""
    a = X.uns["Pf2_A"] * X.uns["Pf2_weights"]
    SNR = a

    # Get the indices for subsetting the data
    sgIndex = X.obs["condition_unique_idxs"]
    Xarr = sps.csr_array(X.X)
    W_proj = np.array(X.obsm["weighted_projections"])

    for i in range(X.uns["Pf2_A"].shape[0]):
        # Parafac2 to slice
        B_i = W_proj[sgIndex == i]
        tslice = np.dot(B_i * a[i], np.array(X.varm["Pf2_C"]).T)

        X_condition_arr = Xarr[sgIndex == i] - X.var["means"].to_numpy()
        err_norm_here = float(np.linalg.norm(X_condition_arr - tslice) ** 2.0)
        SNR[i, :] /= err_norm_here

    return SNR


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


def store_pf2(
    X: anndata.AnnData, parafac2_output: tuple[np.ndarray, list, list]
) -> anndata.AnnData:
    """Store the Pf2 results into the anndata object."""
    sgIndex = X.obs["condition_unique_idxs"]

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = parafac2_output[1]

    X.obsm["projections"] = np.zeros((X.shape[0], len(X.uns["Pf2_weights"])))
    for i, p in enumerate(parafac2_output[2]):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X

def pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
    tolerance=1e-9,
):
    pf_out, _ = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=500
    )

    X = store_pf2(X, pf_out)

    print(f"Degeneracy score: {degeneracy_score((pf_out[0], pf_out[1]))}")

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def pf2_pca_r2x(X: anndata.AnnData, ranks):
    X = X.to_memory()
    XX = sps.csr_array(X.X)

    r2x_pf2 = np.zeros(len(ranks))

    for i in tqdm(range(len(r2x_pf2)), total=len(r2x_pf2)):
        _, R2X = parafac2_nd(X, rank=i+1)
        r2x_pf2[i] = R2X

    pca = PCA(n_components=ranks[-1], svd_solver="arpack")
    pca.fit(XX)
    r2x_pca = np.cumsum(pca.explained_variance_ratio_)

    return r2x_pf2, r2x_pca[np.array(ranks)-1]
