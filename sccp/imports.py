import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.sparse import spmatrix, csr_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis
from .gating import gateThomsonCells


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
) -> anndata.AnnData:
    assert isinstance(X.X, spmatrix)
    assert np.amin(X.X.data) >= 0.0  # type: ignore

    # Filter out genes with too few reads
    readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    X = X[:, readmean > geneThreshold]

    # Normalize read depth
    sc.pp.normalize_total(X, exclude_highly_expressed=False, inplace=True)

    # Scale genes by sum
    readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    readsum = X.shape[0] * readmean
    inplace_column_scale(X.X, 1.0 / readsum)

    # Transform values
    X.X.data = np.log10((1000.0 * X.X.data) + 1.0)  # type: ignore

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    means, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    X.var["means"] = means

    return X


def import_thomson() -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/Thomson/meta.csv", usecols=[0, 1])

    # read in actual data
    # X = sc.read_10x_mtx(
    #     "/opt/andrew/Thomson/", var_names="gene_symbols", make_unique=True
    # )
    # Just immediately written from above line, to make loading faster
    X = anndata.read_h5ad("/opt/andrew/thomson_raw.h5ad")
    obs = X.obs.reset_index(names="cell_barcode")

    # Left merging should put the barcodes in order
    metafile = pd.merge(
        obs, metafile, on="cell_barcode", how="left", validate="one_to_one"
    )

    X.obs = pd.DataFrame(
        {
            "cell_barcode": metafile["cell_barcode"],
            "Condition": pd.Categorical(metafile["sample_id"]),
        }
    )

    doubletDF = pd.read_csv("sccp/data/Thomson/ThomsonDoublets.csv", index_col=0)
    doubletDF.index.name = "cell_barcode"
    X.obs = X.obs.join(doubletDF, on="cell_barcode", how="inner")

    singlet_indices = X.obs.loc[X.obs["doublet"] == 0].index.values
    X.obs = X.obs.reset_index(drop=True)
    X = X[singlet_indices, :]

    X.obs = X.obs.set_index("cell_barcode")
    gateThomsonCells(X)

    return prepare_dataset(X, "Condition", geneThreshold=0.01)


def import_lupus() -> anndata.AnnData:
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
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")
    X = anndata.AnnData(X.raw.X, X.obs, X.raw.var, X.uns, X.obsm)

    protein = anndata.read_h5ad("/opt/andrew/lupus/Lupus_study_protein_adjusted.h5ad")
    protein_df = protein.to_df()

    # rename columns to make more sense
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

    # get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (only 3 cells)
    X = X[X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]

    return prepare_dataset(X, "Condition", geneThreshold=0.1)


def import_citeseq() -> anndata.AnnData:
    """Imports 5 datasets from Hamad CITEseq."""
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                sc.read_10x_mtx,
                "/opt/andrew/HamadCITEseq/" + k,
                gex_only=False,
                make_unique=True,
            )
            for k in files
        ]

        data = {k: futures[i].result() for i, k in enumerate(files)}

    X = anndata.concat(data, merge="same", label="Condition")

    return prepare_dataset(X, "Condition", geneThreshold=0.1)


def import_HTAN() -> anndata.AnnData:
    """Imports Vanderbilt's HTAN 10X data."""
    files = glob.glob("/opt/extra-storage/HTAN/*.mtx.gz")
    futures = []
    data = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        for filename in files:
            future = executor.submit(
                sc.read_10x_mtx,
                "/opt/extra-storage/HTAN/",
                gex_only=False,
                make_unique=True,
                prefix=filename.split("/")[-1].split("matrix.")[0],
            )
            futures.append(future)

        for i, k in enumerate(files):
            result = futures[i].result()
            data[k.split("/")[-1].split("_matrix.")[0]] = result

    X = anndata.concat(data, merge="same", label="Condition")

    return prepare_dataset(X, "Condition", geneThreshold=0.1)


def import_CCLE() -> anndata.AnnData:
    """Imports barcoded cell data."""
    # TODO: Still need to add gene names and barcodes.
    folder = "/opt/extra-storage/asm/Heiser-barcode/CCLE/"

    adatas = {
        "HCT116_1": anndata.read_text(
            Path(folder + "HCT116_tracing_T1.count_mtx.tsv")
        ).T,
        "HCT116_2": anndata.read_text(
            Path(folder + "HCT116_tracing_T2.count_mtx.tsv")
        ).T,
        "MDA-MB-231_1": anndata.read_text(
            Path(folder + "MDA-MB-231_tracing_T1.count_mtx.tsv")
        ).T,
        "MDA-MB-231_2": anndata.read_text(
            Path(folder + "MDA-MB-231_tracing_T2.count_mtx.tsv")
        ).T,
    }

    X = anndata.concat(adatas, label="sample")
    X.X = csr_matrix(X.X)

    return prepare_dataset(X, "sample", geneThreshold=0.1)


def import_cytokine() -> anndata.AnnData:
    """Import Meyer Cytokine PBMC dataset.
    -- columns from observation data:
    {'Stimulation': Cytokine and Dose}
    """
    X = anndata.read_h5ad("/opt/extra-storage/Treg_h5ads/Treg_raw.h5ad")

    # Remove multiplexing identifiers
    X = X[:, ~X.var_names.str.match("^CMO3[0-9]{2}$")]  # type: ignore

    return prepare_dataset(X, "Condition", geneThreshold=0.05)
