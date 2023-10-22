import numpy as np
import pandas as pd
import anndata
import scanpy as sc


def import_thomson() -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/Thomson/meta.csv")

    # Cell barcodes (33482)
    # barcodes = pd.read_csv(
    #     "sccp/data/Thomson/barcodes.tsv", sep="\t", header=None, names=("cell_barcode",)
    # )

    # Left merging should put the barcodes in order
    # metafile = pd.merge(
    #     barcodes, metafile, on="cell_barcode", how="left", validate="one_to_one"
    # )

    # h5ad is simplified version of mtx format
    # import scanpy as sc
    # data = sc.read_10x_mtx("./sccp/data/", var_names='gene_symbols', make_unique=True)
    # data.X = data.X.toarray()
    # data = data[:, np.mean(data.X > 0, axis=0) > 0.001]
    # data.write('thompson.h5ad', compression="gzip")
    X = anndata.read_h5ad("/opt/andrew/thomson.h5ad")

    X.obs["Condition"] = pd.Categorical(metafile["sample_id"])

    X.X /= np.sum(X.X, axis=0)

    # Only operating on the data works because 0 ends up as 0 here
    X.X = np.log10((1000.0 * X.X) + 1)  # scaling factor

    # Center the genes
    X.X -= np.mean(X.X, axis=0)

    return X


def import_lupus() -> anndata.AnnData:
    """Import Lupus PBMC dataset.

    -- columns from observation data:
    {'batch_cov': POOL (1-23) cell was processed in,
    'ind_cov': patient cell was derived from,
    'Processing_Cohort': BATCH (1-4) cell was derived from,
    'louvain': louvain cluster group assignment,
    'cg_cov': broad cell type,
    'ct_cov': lymphocyte-specific cell type,
    'L3': not super clear,
    'ind_cov_batch_cov': combination of patient and pool, proxy for sample ID,
    'Age':	age of patient,
    'Sex': sex of patient,
    'pop_cov': ancestry of patient,
    'Status': SLE status: healthy, managed, treated, or flare,
    'SLE_status': SLE status: healthy or SLE}

    """
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

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

    # get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (only 3 cells)
    X = X[X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]

    # Reorder X so that all of the patients are in alphanumeric order.
    ptIDX = np.argsort(X.obs_vector("Condition"))

    X = X[ptIDX, :]

    # X.X = X.X.toarray()
    # X.write('lupus.h5ad', compression="gzip")
    # X = anndata.read_h5ad("/opt/andrew/thomson.h5ad")

    # Center the genes
    X.X -= np.mean(X.X, axis=0)

    return X


def import_citeseq() -> anndata.AnnData:
    """Imports 5 datasets from Hamad CITEseq."""
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]

    data = {
        k: sc.read_10x_mtx("/opt/andrew/HamadCITEseq/" + k, gex_only=False, make_unique=True) for k in files
    }
    X = anndata.concat(data, merge="same", label="Condition")

    sc.pp.filter_genes(X, min_cells=100)
    sc.pp.normalize_total(X)
    sc.pp.log1p(X)
    sc.pp.highly_variable_genes(X, n_top_genes=4000)

    X = X[:, X.var["highly_variable"]]

    # Center and read normalize the genes
    X.X /= np.sum(X.X, axis=0)
    X.X = X.X.tocsr()

    return X