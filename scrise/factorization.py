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
from parafac2.parafac2 import parafac2_nd, store_pf2
from scipy.stats import gmean
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

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


def pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
    tolerance=1e-6,
    max_iter: int = 100,
    normalize_slices: bool = False,
    backend: str | None = None,
    compress: int | tuple[int, int | None] | str | bool | None = None,
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
        backend=backend,
        compress=compress,
    )

    X = store_pf2(X, pf_out)

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"])

    return X


def rise_pca_r2x(
    X: anndata.AnnData,
    ranks,
    compress: int | tuple[int, int | None] | str | bool | None = "auto",
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
        _, R2X = parafac2_nd(X, rank=i, compress=compress)
        r2x_rise[index] = R2X

    # Mean center because this is done within RISE
    XX = XX.toarray()
    XX = XX - np.mean(XX, axis=0)

    pca = PCA(n_components=ranks[-1])
    pca.fit(XX)
    r2x_pca = np.cumsum(pca.explained_variance_ratio_)

    return r2x_rise, r2x_pca[np.array(ranks) - 1]


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
    if "Pf2_A" not in X.uns or "Pf2_B" not in X.uns or "Pf2_weights" not in X.uns:
        raise KeyError(
            "Input AnnData is missing required uns factors (Pf2_A, Pf2_B, Pf2_weights)."
        )
    if "Pf2_C" not in X.varm:
        raise KeyError("Input AnnData is missing required varm factor 'Pf2_C'.")
    if "projections" not in X.obsm:
        raise KeyError("Input AnnData is missing required obsm 'projections'.")

    # Factor matrices in float32
    uns_dict = {
        k: (
            v.astype(np.float32)
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
            else v
        )
        for k, v in X.uns.items()
    }
    uns_dict["Pf2_A"] = np.asarray(X.uns["Pf2_A"], dtype=np.float32)
    uns_dict["Pf2_B"] = np.asarray(X.uns["Pf2_B"], dtype=np.float32)
    uns_dict["Pf2_weights"] = np.asarray(X.uns["Pf2_weights"], dtype=np.float32)

    varm_dict = {
        k: (
            v.astype(np.float32)
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
            else v
        )
        for k, v in X.varm.items()
    }
    varm_dict["Pf2_C"] = np.asarray(X.varm["Pf2_C"], dtype=np.float32)

    # Compress projections using OPQ
    projections = np.asarray(X.obsm["projections"], dtype=np.float32)
    quantizer, codes, r2 = find_optimal_opq(
        projections,
        fidelity_threshold=fidelity_threshold,
        random_state=random_state,
    )

    assert quantizer.R is not None
    assert quantizer.centroids_cat is not None
    assert quantizer.sub_dims is not None
    uns_dict["opq_rotation"] = quantizer.R.astype(np.float32)
    uns_dict["opq_centroids"] = quantizer.centroids_cat.astype(np.float32)
    uns_dict["opq_subdims"] = quantizer.sub_dims.astype(np.int32)
    uns_dict["opq_fidelity"] = float(r2)

    # Build obsm (excluding weighted_projections and uncompressed projections)
    obsm_dict = {
        "projections_opq_codes": codes.astype(np.uint8),
    }

    # Optionally store PaCMAP embedding
    if "X_pf2_PaCMAP" in X.obsm:
        obsm_dict["embedding"] = np.asarray(X.obsm["X_pf2_PaCMAP"], dtype=np.float32)
    elif "embedding" in X.obsm:
        obsm_dict["embedding"] = np.asarray(X.obsm["embedding"], dtype=np.float32)

    obs_df, var_df = X.obs, X.var
    if not isinstance(obs_df, pd.DataFrame) or not isinstance(var_df, pd.DataFrame):
        raise TypeError(
            "X.obs and X.var must be in-memory pandas DataFrames "
            "(backed Dataset2D is not supported)."
        )
    obs = obs_df.copy()
    # Option B: Compress string barcode index into 2D uint8 ASCII character byte matrix
    orig_index = obs.index.to_numpy(dtype=str)
    max_len = max((len(s) for s in orig_index), default=0)
    if max_len > 0:
        s_arr = orig_index.astype(f"|S{max_len}")
        byte_matrix = np.frombuffer(s_arr.tobytes(), dtype=np.uint8).reshape(
            (len(orig_index), max_len)
        )
        uns_dict["_obs_names_bytes"] = byte_matrix
        obs.index = pd.RangeIndex(len(obs))

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

    # Apply chunked gzip compression to _obs_names_bytes if present
    if "_obs_names_bytes" in uns_dict and len(orig_index) > 1000:
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

    # Restore string barcode index if compressed with Option B
    if "_obs_names_bytes" in adata.uns:
        byte_matrix = np.asarray(adata.uns["_obs_names_bytes"])
        max_len = byte_matrix.shape[1]
        recon_barcodes = np.frombuffer(
            byte_matrix.tobytes(), dtype=f"|S{max_len}"
        ).astype(str)
        adata.obs.index = pd.Index(recon_barcodes)
        del adata.uns["_obs_names_bytes"]

    # Decompress OPQ projections if present
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

    # Reconstruct weighted_projections
    if "projections" in adata.obsm and "Pf2_B" in adata.uns:
        adata.obsm["weighted_projections"] = (
            adata.obsm["projections"].astype(np.float32)
            @ adata.uns["Pf2_B"].astype(np.float32)
        ).astype(np.float32)

    # Restore embedding alias if PaCMAP embedding was stored
    if "embedding" in adata.obsm and "X_pf2_PaCMAP" not in adata.obsm:
        adata.obsm["X_pf2_PaCMAP"] = adata.obsm["embedding"]

    # Optionally match and attach raw data
    if raw_path is not None:
        try:
            import vcsc

            raw = vcsc.VCSCAnnData.read_h5ad(raw_path).to_anndata()
        except (ImportError, AttributeError, KeyError, ValueError, OSError):
            raw = anndata.read_h5ad(raw_path)

        if not isinstance(raw.obs, pd.DataFrame):
            raise TypeError("raw.obs must be an in-memory pandas DataFrame.")

        # Match cells by index or cell_barcode column
        if (
            not np.all(adata.obs_names.isin(raw.obs_names))
            and "cell_barcode" in raw.obs
        ):
            raw.obs.index = pd.Index(raw.obs["cell_barcode"].astype(str))

        # Subset cells present in factors
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
            raise ValueError(
                "No matching gene names found between factors and raw data."
            )
        raw_sub = raw_sub[:, adata.var_names].copy()

        from parafac2.normalize import prepare_dataset

        if "Condition" in adata.obs:
            raw_prep = prepare_dataset(raw_sub, "Condition", geneThreshold=0.0)
            adata.X = raw_prep.X
        else:
            adata.X = raw_sub.X

    return adata
