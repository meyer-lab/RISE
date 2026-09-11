import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import anndata
import h5py
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
from .opq import OPQQuantizer, find_optimal_opq


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

    if X.X is None:
        raise TypeError("X.X must not be None.")
    # X.X's declared type is a large union of array-like/backed-storage types
    # from the AnnData stub; at runtime this is always a dense or sparse
    # in-memory array supporting `.sum`.
    x_count = np.asarray(cast(Any, X.X).sum(axis=1))

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

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
        (and, if present, obsm["weighted_projections"]) reordered/updated
        in place.
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
    X.uns["Pf2_B"] = B[:, order]
    X.uns["Pf2_weights"] = weights[order]

    if "projections" in X.obsm:
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


def _floats_to_float32(mapping) -> dict:
    """Downcast every floating ndarray in an AnnData attribute dict to float32."""
    return {
        k: (
            v.astype(np.float32)
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
            else v
        )
        for k, v in mapping.items()
    }


def _pack_obs_names(obs: pd.DataFrame) -> np.ndarray | None:
    """Replace a string barcode index with a RangeIndex, returning the bytes.

    h5ad stores an object-dtype index far less compactly than a fixed-width
    uint8 matrix, so the barcodes travel in ``uns`` and are rebuilt by
    :func:`_restore_obs_names` on load.
    """
    orig_index = obs.index.to_numpy(dtype=str)
    max_len = max((len(s) for s in orig_index), default=0)
    if max_len == 0:
        return None
    s_arr = orig_index.astype(f"|S{max_len}")
    byte_matrix = np.frombuffer(s_arr.tobytes(), dtype=np.uint8).reshape(
        (len(orig_index), max_len)
    )
    obs.index = pd.RangeIndex(len(obs))
    return byte_matrix


def _recompress_obs_names(filename: str, n_cells: int) -> None:
    """Rewrite the packed barcode matrix chunked and gzipped, once written."""
    if n_cells <= 1000:
        return
    with h5py.File(filename, "r+") as f:
        if "uns/_obs_names_bytes" in f:
            d = f["uns/_obs_names_bytes"][()]
            del f["uns/_obs_names_bytes"]
            f.create_dataset(
                "uns/_obs_names_bytes",
                data=d,
                chunks=(min(16384, len(d)), d.shape[1]),
                compression="gzip",
                compression_opts=6,
            )


def _restore_obs_names(adata: anndata.AnnData) -> None:
    """Rebuild the string barcode index packed by :func:`_pack_obs_names`."""
    if "_obs_names_bytes" not in adata.uns:
        return
    byte_matrix = np.asarray(adata.uns["_obs_names_bytes"])
    max_len = byte_matrix.shape[1]
    recon_barcodes = np.frombuffer(byte_matrix.tobytes(), dtype=f"|S{max_len}").astype(
        str
    )
    adata.obs.index = pd.Index(recon_barcodes)
    del adata.uns["_obs_names_bytes"]


def _restore_projections(adata: anndata.AnnData) -> None:
    """Decode OPQ-compressed projections and rebuild weighted_projections."""
    if "projections_opq_codes" in adata.obsm and "opq_rotation" in adata.uns:
        quantizer = OPQQuantizer.from_saved(
            R=adata.uns["opq_rotation"],
            centroids_cat=adata.uns["opq_centroids"],
            sub_dims=adata.uns["opq_subdims"],
        )
        adata.obsm["projections"] = quantizer.decode(
            np.asarray(adata.obsm["projections_opq_codes"])
        )
    elif "projections" in adata.obsm:
        adata.obsm["projections"] = np.asarray(
            adata.obsm["projections"], dtype=np.float32
        )

    # Never stored on disk: recoverable as projections @ Pf2_B.
    if "projections" in adata.obsm and "Pf2_B" in adata.uns:
        adata.obsm["weighted_projections"] = (
            adata.obsm["projections"].astype(np.float32)
            @ adata.uns["Pf2_B"].astype(np.float32)
        ).astype(np.float32)

    if "embedding" in adata.obsm and "X_pf2_PaCMAP" not in adata.obsm:
        adata.obsm["X_pf2_PaCMAP"] = adata.obsm["embedding"]


def _read_raw_dataset(raw_path: str) -> anndata.AnnData:
    """Read raw expression data, preferring the IVCSR reader when it applies."""
    try:
        import vsparse

        return vsparse.VCSCAnnData.read_h5ad(raw_path).to_anndata()
    except (ImportError, AttributeError, KeyError, ValueError, OSError):
        return anndata.read_h5ad(raw_path)


def _attach_raw_data(adata: anndata.AnnData, raw_path: str) -> None:
    """Match the factors' cells and genes against raw data and attach ``X``."""
    raw = _read_raw_dataset(raw_path)

    if not isinstance(raw.obs, pd.DataFrame):
        raise TypeError("raw.obs must be an in-memory pandas DataFrame.")

    # Match cells by index or cell_barcode column
    if not np.all(adata.obs_names.isin(raw.obs_names)) and "cell_barcode" in raw.obs:
        raw.obs.index = pd.Index(raw.obs["cell_barcode"].astype(str))

    common_cells = adata.obs_names[adata.obs_names.isin(raw.obs_names)]
    if len(common_cells) == 0:
        raise ValueError(
            "No matching cell barcodes found between factors and raw data."
        )
    raw_sub = raw[adata.obs_names, :].copy()

    if not isinstance(raw_sub.var, pd.DataFrame):
        raise TypeError("raw_sub.var must be an in-memory pandas DataFrame.")

    # Match genes
    if (
        not np.all(adata.var_names.isin(raw_sub.var_names))
        and "gene_ids" in raw_sub.var
        and "gene_ids" in adata.var
    ):
        raw_sub.var.index = pd.Index(raw_sub.var["gene_ids"].astype(str))

    common_genes = adata.var_names[adata.var_names.isin(raw_sub.var_names)]
    if len(common_genes) == 0:
        raise ValueError("No matching gene names found between factors and raw data.")
    raw_sub = raw_sub[:, adata.var_names].copy()

    from parafac2.normalize import prepare_dataset

    if "Condition" in adata.obs:
        raw_prep = prepare_dataset(raw_sub, "Condition", geneThreshold=0.0)
        adata.X = raw_prep.X
    else:
        adata.X = raw_sub.X


def _require_export_factors(X: anndata.AnnData) -> None:
    """Reject an AnnData that has not been through a RISE fit."""
    if "Pf2_A" not in X.uns or "Pf2_B" not in X.uns or "Pf2_weights" not in X.uns:
        raise KeyError(
            "Input AnnData is missing required uns factors (Pf2_A, Pf2_B, Pf2_weights)."
        )
    if "Pf2_C" not in X.varm:
        raise KeyError("Input AnnData is missing required varm factor 'Pf2_C'.")
    if "projections" not in X.obsm:
        raise KeyError("Input AnnData is missing required obsm 'projections'.")


def _compress_projections(
    X: anndata.AnnData, fidelity_threshold: float, random_state: int
) -> tuple[np.ndarray, dict]:
    """OPQ-quantize the projections, returning the codes and the codebook."""
    projections = np.asarray(X.obsm["projections"], dtype=np.float32)
    quantizer, codes, r2 = find_optimal_opq(
        projections,
        fidelity_threshold=fidelity_threshold,
        random_state=random_state,
    )

    assert quantizer.R is not None
    assert quantizer.centroids_cat is not None
    assert quantizer.sub_dims is not None
    return codes, {
        "opq_rotation": quantizer.R.astype(np.float32),
        "opq_centroids": quantizer.centroids_cat.astype(np.float32),
        "opq_subdims": quantizer.sub_dims.astype(np.int32),
        "opq_fidelity": float(r2),
    }


def _embedding_for_export(X: anndata.AnnData) -> dict:
    """The PaCMAP embedding under its on-disk name, if the fit produced one."""
    for key in ("X_pf2_PaCMAP", "embedding"):
        if key in X.obsm:
            return {"embedding": np.asarray(X.obsm[key], dtype=np.float32)}
    return {}


def export_factors(
    X: anndata.AnnData,
    filename: str,
    fidelity_threshold: float = 0.99,
    random_state: int = 42,
) -> anndata.AnnData:
    """Export RISE decomposition factors to an h5ad file without raw expression data.

    Compresses the projection matrix using Optimized Product Quantization (OPQ)
    to meet or exceed the specified fidelity threshold (R^2 >= fidelity_threshold).
    All factor matrices (Pf2_A, Pf2_B, Pf2_weights, Pf2_C) are stored in float32.
    Weighted projections are never stored on disk because they can be reconstructed
    deterministically as projections @ Pf2_B.
    PaCMAP embeddings are optionally stored if present.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData object containing RISE decomposition results. Must contain:
        - X.uns["Pf2_A"], X.uns["Pf2_B"], X.uns["Pf2_weights"]
        - X.varm["Pf2_C"]
        - X.obsm["projections"]
    filename : str
        Output file path (.h5ad).
    fidelity_threshold : float, optional (default: 0.99)
        Target R^2 reconstruction accuracy threshold for projection compression.
    random_state : int, optional (default: 42)
        Random seed for reproducibility during OPQ codebook training.

    Returns
    -------
    anndata.AnnData
        The factor-only AnnData object written to disk.
    """
    _require_export_factors(X)

    # Factor matrices in float32
    uns_dict = _floats_to_float32(X.uns)
    uns_dict["Pf2_A"] = np.asarray(X.uns["Pf2_A"], dtype=np.float32)
    uns_dict["Pf2_B"] = np.asarray(X.uns["Pf2_B"], dtype=np.float32)
    uns_dict["Pf2_weights"] = np.asarray(X.uns["Pf2_weights"], dtype=np.float32)

    varm_dict = _floats_to_float32(X.varm)
    varm_dict["Pf2_C"] = np.asarray(X.varm["Pf2_C"], dtype=np.float32)

    codes, opq_uns = _compress_projections(X, fidelity_threshold, random_state)
    uns_dict.update(opq_uns)

    # obsm excludes weighted_projections and the uncompressed projections.
    obsm_dict = {"projections_opq_codes": codes.astype(np.uint8)}
    obsm_dict.update(_embedding_for_export(X))

    obs_df, var_df = X.obs, X.var
    if not isinstance(obs_df, pd.DataFrame) or not isinstance(var_df, pd.DataFrame):
        raise TypeError(
            "X.obs and X.var must be in-memory pandas DataFrames "
            "(backed Dataset2D is not supported)."
        )
    obs = obs_df.copy()
    n_cells = len(obs)
    packed_names = _pack_obs_names(obs)
    if packed_names is not None:
        uns_dict["_obs_names_bytes"] = packed_names

    factors_adata = anndata.AnnData(
        obs=obs,
        var=var_df.copy(),
        uns=uns_dict,
        varm=cast(Mapping[str, Sequence[Any]], varm_dict),
        obsm=cast(Mapping[str, Sequence[Any]], obsm_dict),
    )

    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    factors_adata.write_h5ad(filename)

    if "_obs_names_bytes" in uns_dict:
        _recompress_obs_names(filename, n_cells)
    return factors_adata


def load_factors(
    filename: str,
    raw_path: str | None = None,
) -> anndata.AnnData:
    """Load RISE decomposition factors from an h5ad file, decompressing OPQ projections
    and optionally rebuilding the full dataset from raw expression data.

    Parameters
    ----------
    filename : str
        Path to factors .h5ad file.
    raw_path : str, optional
        Path to raw AnnData or IVCSR .h5ad / .h5 file. If provided, cells and genes
        are matched based on cell barcodes and gene names to attach the expression matrix.

    Returns
    -------
    anndata.AnnData
        AnnData object with reconstructed projections, weighted_projections,
        factors, and optionally the raw expression matrix X.
    """
    adata = anndata.read_h5ad(filename)
    _restore_obs_names(adata)
    _restore_projections(adata)

    if raw_path is not None:
        _attach_raw_data(adata, raw_path)

    return adata
