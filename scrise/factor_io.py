"""Reading and writing RISE factors, without the raw expression matrix.

Factors are exported to h5ad with the projection matrix OPQ-quantized and
the cell barcodes packed into a uint8 matrix, both of which have to be
undone on load.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import anndata
import h5py
import hdf5plugin  # noqa: F401  (registers the HDF5 filters these files use)
import numpy as np
import pandas as pd

from .opq import OPQQuantizer, find_optimal_opq


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
    """Replace a string barcode index with a RangeIndex, returning the bytes."""
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
