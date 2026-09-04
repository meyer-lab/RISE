import os
import urllib.request

import anndata
import h5py
import pandas as pd
import vsparse
from anndata.io import read_elem
from parafac2.normalize import prepare_dataset

THOMSON_RAW_URL = (
    "https://ucla.box.com/shared/static/jy53rcort51xn5t2dr5927wfj13g9e7j.h5"
)


def download_thomson_raw(
    dest_path: str = "analysis/data/Thomson/thomson_raw.h5ad",
) -> str:
    """Download the raw Thomson IVCSR dataset if not present locally."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(THOMSON_RAW_URL, dest_path)
    return dest_path


def import_thomson() -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    from .gating import gateThomsonCells

    raw_path = download_thomson_raw()
    X = vsparse.VCSCAnnData.read_h5ad(raw_path).to_anndata()

    doubletDF = pd.read_csv("analysis/data/Thomson/ThomsonDoublets.csv", index_col=0)
    doubletDF.index.name = "cell_barcode"
    X.obs = X.obs.join(doubletDF, how="inner")

    singlet_indices = X.obs.loc[X.obs["doublet"] == 0].index.values
    X = X[singlet_indices, :]

    gateThomsonCells(X)

    return prepare_dataset(X, "Condition", geneThreshold=0.01)


def import_thomson_factors(include_raw: bool = False) -> anndata.AnnData:
    """Import Thomson dataset with cached PARAFAC2 factors and projections."""
    from scrise import load_factors

    if include_raw:
        raw_path = download_thomson_raw()
        return load_factors(
            "analysis/data/Thomson_cached_factors.h5ad", raw_path=raw_path
        )
    return load_factors("analysis/data/Thomson_cached_factors.h5ad")


def import_lupus(geneThreshold: float = 0.1) -> anndata.AnnData:
    """Import Lupus PBMC dataset.

    -- columns from observation data:
    {'batch_cov': POOL (1-23) cell was processed in,
    'ind_cov': patient cell was derived from,
    'Processing_Cohort': BATCH (1-4) cell was derived from,
    'louvain': louvain cluster group assignment,
    'cg_cov': broad cell type,
    'ct_cov': lymphocyte-specific cell type,
    'L3': marks a balanced subset of batch 4 used for model training,
    'ind_cov_batch_cov': combination of patient and pool, proxy for sample ID,
    'Age':	age of patient,
    'Sex': sex of patient,
    'pop_cov': ancestry of patient,
    'Status': SLE status: healthy, managed, treated, or flare,
    'SLE_status': SLE status: healthy or SLE}

    """
    # Read only obs, raw/X, and raw/var — the dense processed X slot (9.6 GB float32)
    # causes hdf5plugin to allocate ~180 GB of decompression buffers when the file is
    # opened, even in backed mode. h5py lets us skip it entirely.
    with h5py.File("/opt/andrew/lupus/lupus.h5ad", "r") as f:
        obs = read_elem(f["obs"])
        raw_var = read_elem(f["raw/var"])
        raw_X = read_elem(f["raw/X"])

    X = anndata.AnnData(X=raw_X, obs=obs, var=raw_var)

    protein = anndata.read_h5ad("/opt/andrew/lupus/Lupus_study_protein_adjusted.h5ad")
    protein_df = protein.to_df()

    # Rename columns
    X.obs = X.obs.rename(
        {
            "batch_cov": "pool",
            "ind_cov": "patient",
            "cg_cov": "Cell Type",
            "ct_cov": "cell_type_lympho",
            "ind_cov_batch_cov": "Condition",
            "Age": "age",
            "Sex": "sex",
            "pop_cov": "ancestry",
        },
        axis=1,
    )

    X.obs = X.obs.merge(protein_df, how="left", left_index=True, right_index=True)

    # Get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (Only 3 cells)
    # .copy() materialises the boolean-mask view before prepare_dataset; without it,
    # prepare_dataset's `X.X = csr_array(X.X)` triggers a sparse setitem on the parent
    # matrix (parent._X[bool_array, :] = value) which forces scipy to convert the
    # 1.26M×32738 CSR to LIL/dense → OOM.
    mask = X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"
    return prepare_dataset(X[mask].copy(), "Condition", geneThreshold=geneThreshold)
