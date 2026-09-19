from typing import Any, cast

import anndata
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import scipy.sparse as sps
from pacmap import PaCMAP
from parafac2.parafac2 import store_pf2
from scipy.stats import gmean
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from ._pf2_utils import run_parafac2


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
    sgIndex = np.asarray(X.obs["condition_unique_idxs"])
    cond_mean = gmean(X.uns["Pf2_A"], axis=1)

    if X.X is None:
        raise TypeError("X.X must not be None.")
    # X.X's declared type is a large union of array-like/backed-storage types
    # from the AnnData stub; at runtime this is always a dense or sparse
    # in-memory array supporting `.sum`.
    x_count = np.asarray(cast(Any, X.X).sum(axis=1)).ravel()

    n_conds = int(np.amax(sgIndex)) + 1
    counts = np.bincount(sgIndex, weights=x_count, minlength=n_conds).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct


def canonical_component_signs(C: np.ndarray) -> np.ndarray:
    """Compute a canonical sign for each component of a gene factor matrix.

    PARAFAC2 components are only identified up to a sign flip that is shared
    between two of the three factor matrices (Harshman's Uniqueness Theorem
    fixes the decomposition up to permutation and sign/scale). This makes any
    comparison across fits (different ranks, different random seeds, etc.)
    ambiguous unless a canonical sign is first imposed. We adopt the
    convention that the largest-magnitude entry of each gene-factor (``C``)
    column should be positive.

    Parameters
    ----------
    C : numpy.ndarray
        Gene factor matrix, shape (n_genes, rank).

    Returns
    -------
    numpy.ndarray
        Array of +1/-1 signs, shape (rank,), one per component.
    """
    rank = C.shape[1]
    max_idx = np.argmax(np.abs(C), axis=0)
    signs = np.ones(rank)
    signs[C[max_idx, np.arange(rank)] < 0] = -1.0
    return signs


def order_components_by_energy(X: anndata.AnnData) -> anndata.AnnData:
    """Reorder PARAFAC2 components by intrinsic energy.

    RISE previously inherited an ordering of components based on the Gini
    coefficient (variance-to-mean ratio) of the condition factor. That
    ordering is a heuristic: it is not particularly stable, and it does not
    guarantee that, when moving from rank N to rank N+1, the first N
    components of the new fit correspond to the N components of the old fit.

    This function instead orders components by their intrinsic energy,
    ``|weights[r]| * ||A[:, r]|| * ||C[:, r]||`` (the product of component
    weights, condition-factor, and gene-factor column norms), which is
    directly determined by the fit and is not subject to arbitrary rescaling.
    Components are ordered from highest to lowest energy, so that low-energy
    components -- which tend to be the ones added when the rank is
    increased -- land at the high end of the ordering. This is consistent
    with the expectation, motivated by Harshman's Uniqueness Theorem for
    PARAFAC2 (which fixes the decomposition up to permutation and
    sign/scale given >= 3 conditions and full-column-rank factors), that a
    rank-(N+1) fit should mostly agree with a rank-N fit on its first N
    components.

    Before computing the ordering, a canonical sign is imposed on each
    component (see :func:`canonical_component_signs`) so that the ordering,
    and any downstream comparison across fits, is well defined.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData object containing RISE decomposition results (as produced by
        :func:`pf2`). Must contain X.uns["Pf2_A"], X.uns["Pf2_B"],
        X.uns["Pf2_weights"], and X.varm["Pf2_C"].

    Returns
    -------
    anndata.AnnData
        The same AnnData object, with Pf2_A, Pf2_B, Pf2_C, Pf2_weights
        (and, if present, obsm["projections"] and
        obsm["weighted_projections"]) reordered/updated in place. The
        eigen-state axis of Pf2_B, and the matching columns of
        obsm["projections"], are permuted alongside the components so that
        the maximal-diagonal form of B established by
        ``parafac2.utils.standardize_pf2`` is preserved.
    """
    A = np.array(X.uns["Pf2_A"])
    B = np.array(X.uns["Pf2_B"])
    C = np.array(X.varm["Pf2_C"])
    weights = np.array(X.uns["Pf2_weights"])

    # Canonical sign convention, shared between the condition factor (A) and
    # the gene factor (C); the eigen-state factor (B) is left as the
    # unflipped reference so that the product A x B x C is unchanged.
    signs = canonical_component_signs(C)
    A = A * signs
    C = C * signs

    energy = np.abs(weights) * np.linalg.norm(A, axis=0) * np.linalg.norm(C, axis=0)
    order = np.argsort(energy)[::-1]

    X.uns["Pf2_A"] = A[:, order]
    X.varm["Pf2_C"] = C[:, order]
    X.uns["Pf2_weights"] = weights[order]

    # B is indexed by (eigen-state, component). ``parafac2.utils.standardize_pf2``
    # permutes B's *rows* so that its diagonal is maximal, pairing eigen-state i
    # with component i, and permutes the columns of each projection to match.
    # Permuting only B's columns here would move each diagonal entry off the
    # diagonal and destroy that pairing, so the eigen-state axis is relabeled by
    # the same permutation. Since P_k B is left unchanged by this relabeling
    # (up to the component permutation), the reconstruction is untouched.
    X.uns["Pf2_B"] = B[np.ix_(order, order)]

    if "projections" in X.obsm:
        X.obsm["projections"] = np.asarray(X.obsm["projections"])[:, order]
        X.obsm["weighted_projections"] = (
            X.obsm["projections"] @ X.uns["Pf2_B"]
        ).astype(np.float32, copy=False)

    return X


def match_components_across_ranks(
    C_low: np.ndarray, C_high: np.ndarray, threshold: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """Match components between two PARAFAC2 fits of adjacent rank by cosine
    similarity of their gene factors.

    This provides the "does component X at rank N correspond to component Y
    at rank N-1" matching primitive described in issue #520. It is a small,
    self-contained addition, not a full cross-rank benchmarking pipeline:
    given the gene factors of a rank-N fit and a rank-(N+1) fit (each
    already sign-canonicalized, e.g. via :func:`canonical_component_signs`),
    it performs Hungarian maximum-weight matching on cosine similarity and
    reports which components matched, and which rank-(N+1) component(s) had
    no good match (i.e. are candidates for the newly added component).

    Parameters
    ----------
    C_low : numpy.ndarray
        Gene factor matrix of the lower-rank fit, shape (n_genes, rank_low).
    C_high : numpy.ndarray
        Gene factor matrix of the higher-rank fit, shape (n_genes,
        rank_high), with rank_high >= rank_low.
    threshold : float, optional (default: 0.6)
        Minimum cosine similarity for two components to be considered a
        match.

    Returns
    -------
    tuple of numpy.ndarray
        matched_pairs : numpy.ndarray of shape (n_matches, 2)
            Each row is (index in C_low, index in C_high) for a matched
            pair of components.
        unmatched_high : numpy.ndarray
            Indices into C_high of components with no match above
            ``threshold`` -- candidates for the newly added component(s).
    """
    from scipy.optimize import linear_sum_assignment

    C_low_n = C_low / np.linalg.norm(C_low, axis=0, keepdims=True)
    C_high_n = C_high / np.linalg.norm(C_high, axis=0, keepdims=True)

    cos_sim = C_low_n.T @ C_high_n  # (rank_low, rank_high)

    row_ind, col_ind = linear_sum_assignment(-cos_sim)

    matched_mask = cos_sim[row_ind, col_ind] >= threshold
    matched_pairs = np.stack([row_ind[matched_mask], col_ind[matched_mask]], axis=1)

    unmatched_high = np.setdiff1d(
        np.arange(C_high.shape[1]), matched_pairs[:, 1] if matched_pairs.size else []
    )

    return matched_pairs, unmatched_high


def pf2(
    X: anndata.AnnData | None = None,
    rank: int | None = None,
    random_state=1,
    doEmbedding: bool = True,
    tolerance=1e-6,
    max_iter: int = 100,
    normalize_slices: bool = False,
    backend: str | None = None,
    compress: int | tuple[int, int | None] | str | bool | None = None,
    compression_kwarg: dict[str, Any] | None = None,
    parafac2_kwarg: dict[str, Any] | None = None,
    condition_key: str | None = None,
    adata: anndata.AnnData | None = None,
) -> anndata.AnnData:
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
    tolerance : float, optional (default: 1e-6)
        Convergence threshold for the optimization algorithm. Lower values
        increase precision but may require more iterations.
    max_iter : int, optional (default: 100)
        Maximum number of iterations for the optimization algorithm.
    normalize_slices : bool, optional (default: False)
        If True, normalizes per-condition slices by their Frobenius norm during
        factor updates to prevent conditions with large cell counts from dominating.
    backend : str | None, optional (default: None)
        Compute backend to run matrix products on: one of ``'mlx'``, ``'cupy'``,
        or ``'cpu'``. If None, the first available accelerator is auto-detected
        (see :func:`~parafac2.backend.get_backend`).
    compress : int | tuple[int, int | None] | str | bool | None, optional (default: None)
        CANDELINC compression mode passed to ``parafac2_nd``. If None/False
        (default), exact ALS is used. If ``"auto"`` or True, compression
        dimensions are set automatically from ``rank``. See
        :func:`parafac2.parafac2.parafac2_nd` for details.
    compression_kwarg : dict, optional
        Additional keyword arguments forwarded to
        :func:`parafac2.compress.compress_dataset` (e.g. ``n_power_iter``).
        Requires ``compress`` to also be set.
    parafac2_kwarg : dict, optional
        Additional keyword arguments forwarded to ``parafac2_nd`` (e.g.
        ``n_inner``, ``callback``), for options not otherwise exposed here.

    condition_key : str, optional (default: None)
        Column in ``X.obs`` holding the condition labels, used to derive
        ``condition_unique_idxs`` when that column is not already present.
    adata : anndata.AnnData, optional (default: None)
        Alias for ``X``; supply either one, not both.

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

        Components are ordered from highest to lowest intrinsic energy
        (see :func:`order_components_by_energy`), so that low-energy
        components -- typically the ones added when the rank is increased
        -- land at the high end of the ordering.
    """
    if X is None and adata is not None:
        X = adata
    if X is None:
        raise ValueError("Either X or adata must be provided.")
    if rank is None:
        raise ValueError("rank must be provided.")

    if "condition_unique_idxs" not in X.obs:
        if condition_key is not None and condition_key in X.obs:
            X.obs["condition_unique_idxs"] = pd.Categorical(X.obs[condition_key]).codes
        else:
            raise KeyError(
                "X.obs must contain 'condition_unique_idxs', or provide 'condition_key' pointing to a valid column in X.obs."
            )

    pf_out, _ = run_parafac2(
        X,
        rank=rank,
        random_state=random_state,
        tol=tolerance,
        n_iter_max=max_iter,
        normalize_slices=normalize_slices,
        backend=backend,
        compress=compress,
        compression_kwarg=compression_kwarg,
        parafac2_kwarg=parafac2_kwarg,
    )

    X = store_pf2(X, pf_out)
    X = order_components_by_energy(X)

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])

    return X


def rise_pca_r2x(
    X: anndata.AnnData,
    ranks,
    compress: int | tuple[int, int | None] | str | bool | None = "auto",
    compression_kwarg: dict[str, Any] | None = None,
    parafac2_kwarg: dict[str, Any] | None = None,
):
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
    compress : int | tuple[int, int | None] | str | bool | None, optional
        CANDELINC compression mode passed to ``parafac2_nd`` for each rank's
        fit. Defaults to ``"auto"`` (compression dimensions set from each
        rank), which sharply cuts the cost of sweeping many ranks over raw
        data. Pass None/False to fall back to exact ALS.
    compression_kwarg : dict, optional
        Additional keyword arguments forwarded to
        :func:`parafac2.compress.compress_dataset` (e.g. ``n_power_iter``).
        Requires ``compress`` to also be set.
    parafac2_kwarg : dict, optional
        Additional keyword arguments forwarded to ``parafac2_nd`` for each
        rank's fit (e.g. ``normalize_slices``, ``backend``, ``tol``).

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
        _, R2X = run_parafac2(
            X,
            rank=i,
            random_state=None,
            tol=1e-6,
            n_iter_max=100,
            compress=compress,
            compression_kwarg=compression_kwarg,
            parafac2_kwarg=parafac2_kwarg,
        )
        r2x_rise[index] = R2X

    # Mean center because this is done within RISE
    XX = XX.toarray()
    XX = XX - np.mean(XX, axis=0)

    pca = PCA(n_components=ranks[-1])
    pca.fit(XX)
    r2x_pca = np.cumsum(pca.explained_variance_ratio_)

    return r2x_rise, r2x_pca[np.array(ranks) - 1]


__all__ = [
    "canonical_component_signs",
    "correct_conditions",
    "match_components_across_ranks",
    "order_components_by_energy",
    "pf2",
    "rise_pca_r2x",
]
