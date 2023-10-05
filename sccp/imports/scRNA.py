import numpy as np
import pandas as pd
from scipy.stats import linregress
import anndata
from ..parafac2 import Pf2X


def tensorFy(annD: anndata.AnnData, obsName: str) -> Pf2X:
    obsV = annD.obs_vector(obsName)
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    data = [annD[sgIndex == sgi, :].X.toarray() for sgi in range(len(sgUnique))]

    return Pf2X(data, sgUnique, annD.var_names)


def import_perturb_RPE():
    ds_disk = anndata.read_h5ad("/opt/andrew/rpe1_normalized_singlecell_01.h5ad")

    # Remove NaNs
    ds_disk = ds_disk[:, np.all(np.isfinite(ds_disk.X), axis=0)]

    return tensorFy(ds_disk, "sgID_AB")


def import_thompson_drug() -> anndata.AnnData:
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper."""

    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/Thomson/meta.csv")

    # Cell barcodes (33482)
    barcodes = pd.read_csv(
        "sccp/data/Thomson/barcodes.tsv", sep="\t", header=None, names=("cell_barcode",)
    )

    # Left merging should put the barcodes in order
    metafile = pd.merge(
        barcodes, metafile, on="cell_barcode", how="left", validate="one_to_one"
    )

    # h5ad is simplified version of mtx format
    # import scanpy as sc
    # data = sc.read_10x_mtx("./sccp/data/", var_names='gene_symbols', make_unique=True)
    # data.X = data.X.todense()
    # data = data[:, np.mean(data.X > 0, axis=0) > 0.001]
    # data.write('thompson.h5ad', compression="gzip")
    data = anndata.read_h5ad("/opt/andrew/thomson.h5ad")

    data.obs["Drugs"] = pd.Categorical(metafile["sample_id"])
    return data


def ThompsonXA_SCGenes(offset: float = 1.0) -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    X = import_thompson_drug()
    scalingfactor = 1000

    assert np.all(np.isfinite(X.X.data))

    X = X[:, np.mean(X.X > 0, axis=0) > 0.001]
    X.X /= np.sum(X.X, axis=0)

    # Only operating on the data works because 0 ends up as 0 here
    X.X = np.log10((scalingfactor * X.X) + 1)
    means = np.mean(X.X, axis=0)
    cv = np.std(X.X, axis=0) / means

    logmean = np.log10(means + 1e-10)
    logstd = np.log10(cv + 1e-10)

    if offset != 1.0:
        slope, intercept, _, _, _ = linregress(logmean, logstd)

        above_idx = logstd > logmean * slope + intercept + np.log10(offset)
        X = X[:, above_idx]

    # Center the genes
    X.X -= np.mean(X.X, axis=0)

    # Assign cells a count per-experiment so we can reindex
    return tensorFy(X, "Drugs")


def load_lupus_data():
    """Import Lupus PBMC dataset.

    *NOTE*: This function has two outputs, not one. The first is the data in tensor format,
    the second is the 'observations' anndata associated data (a pandas DataFrame)

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
            "cg_cov": "cell_type_broad",
            "ct_cov": "cell_type_lympho",
            "ind_cov_batch_cov": "sample_ID",
            "Age": "age",
            "Sex": "sex",
            "pop_cov": "ancestry",
            "Status": "SLE_condition",
        },
        axis=1,
    )

    # get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (only 3 cells)
    X = X[X.obs["sample_ID"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]

    # reorder X so that all of the patients are in alphanumeric order. this is important
    # so that we can steal cell typings at this point
    obsV = X.obs_vector("sample_ID")
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    ann_data_objects = [X[sgIndex == sgi, :] for sgi in range(len(sgUnique))]

    X = anndata.concat(ann_data_objects, axis=0)

    assert np.all(np.isfinite(X.X.data))  # this should be true

    return tensorFy(X, "sample_ID"), X.obs
