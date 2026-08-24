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
    """Correct the condition factors by normalizing for overall read depth.

    This function adjusts condition factors (stored in X.uns["Pf2_A"]) to account for
    differences in sequencing depth across conditions. It uses linear regression to
    model the relationship between total read counts and condition factor magnitudes,
    then applies a correction.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData object containing RISE decomposition results. Must have:
        - X.obs["condition_unique_idxs"]: 0-indexed condition assignments
        - X.uns["Pf2_A"]: Condition factors from PARAFAC2 decomposition

    Returns
    -------
    numpy.ndarray
        Corrected condition factors normalized by sequencing depth
    """
    sgIndex = X.obs["condition_unique_idxs"]

    counts = np.zeros((np.amax(sgIndex.to_numpy()) + 1, 1))

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
    max_iter: int = 500,
    normalize_slices: bool = False,
):
    """Perform PARAFAC2 tensor decomposition on single-cell RNA-seq data.

    This is the main function for running RISE analysis. It decomposes the
    multi-condition single-cell data into condition factors, eigen-state factors,
    and gene factors, revealing patterns across experimental conditions.

    Parameters
    ----------
    X : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] indicating which condition
        each cell belongs to (0-indexed).
    rank : int
        Number of components to extract. Determines the complexity of the
        decomposition. Typically chosen based on variance explained and
        Factor Match Score analysis (see plot_r2x and plot_fms_diff_ranks).
    random_state : int, optional (default: 1)
        Random seed for reproducibility of the decomposition.
    doEmbedding : bool, optional (default: True)
        If True, automatically computes PaCMAP embedding of cell projections
        and stores in X.obsm["X_pf2_PaCMAP"]. This enables visualization
        functions like plot_labels_pacmap.
    tolerance : float, optional (default: 1e-9)
        Convergence threshold for the optimization algorithm. Lower values
        increase precision but may require more iterations.
    max_iter : int, optional (default: 500)
        Maximum number of iterations for the optimization algorithm.
    normalize_slices : bool, optional (default: False)
        If True, normalizes per-condition slices by their Frobenius norm during
        factor updates to prevent conditions with large cell counts from dominating.

    Returns
    -------
    anndata.AnnData
        The input AnnData object with added RISE decomposition results:

        - X.uns["Pf2_weights"]: Component weights (shape: rank,)
        - X.uns["Pf2_A"]: Condition factors (shape: n_conditions, rank)
        - X.uns["Pf2_B"]: Eigen-state factors (shape: rank, rank)
        - X.varm["Pf2_C"]: Gene factors (shape: n_genes, rank)
        - X.obsm["projections"]: Cell projections (shape: n_cells, rank)
        - X.obsm["weighted_projections"]: Weighted cell projections
          (shape: n_cells, rank)
        - X.obsm["X_pf2_PaCMAP"]: PaCMAP embedding (shape: n_cells, 2)
          if doEmbedding=True
    """
    pf_out, _ = parafac2_nd(
        X,
        rank=rank,
        random_state=random_state,
        tol=tolerance,
        n_iter_max=max_iter,
        normalize_slices=normalize_slices,
    )

    X = store_pf2(X, pf_out)

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def rise_pca_r2x(X: anndata.AnnData, ranks):
    """Compute variance explained (R²X) for RISE and PCA across different ranks.

    This function evaluates how much variance in the data is explained by
    RISE (PARAFAC2) and PCA decompositions at different component ranks.
    Used to determine the optimal number of components for RISE analysis.

    Parameters
    ----------
    X : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] for RISE decomposition.
    ranks : array-like of int
        Array of rank values to test (e.g., [1, 5, 10, 15, 20, 25, 30]).
        Each rank represents a different number of components.

    Returns
    -------
    tuple of numpy.ndarray
        (rise_r2x, pca_r2x) where:

        - rise_r2x: Variance explained by RISE for each rank (shape: len(ranks),)
        - pca_r2x: Variance explained by PCA for each rank (shape: len(ranks),)
    """
    X = X.to_memory()
    XX = sps.csr_array(X.X)

    r2x_rise = np.zeros(len(ranks))

    for index, i in tqdm(enumerate(ranks), total=len(r2x_rise)):
        _, R2X = parafac2_nd(X, rank=i)
        r2x_rise[index] = R2X

    # Mean center because this is done within RISE
    XX = XX.toarray()
    XX = XX - np.mean(XX, axis=0)

    pca = PCA(n_components=ranks[-1])
    pca.fit(XX)
    r2x_pca = np.cumsum(pca.explained_variance_ratio_)

    return r2x_rise, r2x_pca[np.array(ranks) - 1]
