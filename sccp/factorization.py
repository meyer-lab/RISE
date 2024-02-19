from pacmap import PaCMAP
import scipy.sparse as sps
from tensorly.cp_tensor import CPTensor
from sklearn.linear_model import LinearRegression
from scipy.stats import gmean
from tlviz.factor_tools import (
    factor_match_score as fms,
    degeneracy_score,
)
from parafac2.parafac2 import parafac2_nd


import anndata
import numpy as np
from tqdm import tqdm


def cwSNR(
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


def pf2_r2x(
    X: anndata.AnnData,
    max_rank: int,
) -> np.ndarray:
    X = X.to_memory()

    r2x_vec = np.empty(max_rank)

    for i in tqdm(range(len(r2x_vec)), total=len(r2x_vec)):
        _, R2X = parafac2_nd(
            X,
            rank=i + 1,
        )

        r2x_vec[i] = R2X

    return r2x_vec


def pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
    tolerance=1e-10,
):
    pf_out, _ = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=1000
    )

    X = store_pf2(X, pf_out)

    print(f"Degeneracy score: {degeneracy_score((pf_out[0], pf_out[1]))}")

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def pf2_fms(
    X: anndata.AnnData,
    max_rank: int,
    random_state=1,
) -> np.ndarray:
    # Get the indices for subsetting the data
    indices = np.arange(0, X.shape[0])

    rng1 = np.random.default_rng(random_state)
    indices1 = rng1.choice(indices, size=X.shape[0], replace=True)

    X1 = X[indices1, :].to_memory()
    X2 = X.to_memory()

    fms_vec = np.empty(max_rank)

    for i in tqdm(range(len(fms_vec)), total=len(fms_vec)):
        parafac2_output1, _ = parafac2_nd(
            X1,
            rank=i + 1,
        )
        parafac2_output2, _ = parafac2_nd(
            X2,
            rank=i + 1,
        )

        X1cp = CPTensor((parafac2_output1[0], parafac2_output1[1]))
        X2cp = CPTensor((parafac2_output2[0], parafac2_output2[1]))

        fms_vec[i] = fms(X1cp, X2cp, consider_weights=True, skip_mode=None)

    return fms_vec
