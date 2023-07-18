from os.path import dirname
import numpy as np
import pandas as pd
from scipy.stats import linregress
import anndata
from ..parafac2 import Pf2X


path_here = dirname(dirname(__file__))


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
    metafile = pd.read_csv("sccp/data/meta.csv")

    # Cell barcodes (33482)
    barcodes = pd.read_csv(
        "sccp/data/barcodes.tsv", sep="\t", header=None, names=("cell_barcode",)
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
    data = anndata.read_h5ad("/opt/andrew/thompson.h5ad")

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


def import_pancreas(tensor=True, method=str()):
    pancreas = anndata.read_h5ad(
        "/home/brianoj/SC_data/pancreas/pancreas" + method + ".h5ad"
    )

    # Remove NaNs
    pancreas = pancreas[:, np.all(np.isfinite(pancreas.X), axis=0)]

    if tensor:
        return tensorFy(pancreas, "batch")
    else:
        return pancreas


def import_pancreas_all(tensor=True, method=str()):
    pancreas = anndata.read_h5ad("/home/brianoj/SC_data/pancreas/pancreas.h5ad")
    pancreas.obsm["Unintegrated"] = pancreas.obsm["X_pca"]
    methods = ["Unintegrated", "scanorama", "mnnpy", "mnncorrect", "harmony", "cca", "bbknn_trim", "bbknn_faiss", "bbknn_ckdtree", "bbknn"]
    for method in methods:
        pancreas_corr = anndata.read_h5ad("/home/brianoj/SC_data/pancreas/pancreas_" + method + ".h5ad")
        pancreas.obsm[method] = pancreas_corr.obsm["X_pca"]

    return pancreas, methods


def load_lupus_data(third_axis="ind_cov", every_n=4):
    """Import Lupus PBMC dataset.

    `third_axis`: 3rd dimension along with to expand data. Defaults to patient (ind_cov)
    `every_n`: takes every nth cell to be included. set to 1 to include all data

    *NOTE*: This function has three outputs, not one. The first is the data in tensor format,
    the second is a list of cell types compatible with later functions that plot by cell type,
    and the third is a list of row colors for use coloring factor plot A by SLE status.
    """
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

    # reorder X so that all of the patients are in alphanumeric order. this is important
    # so that we can steal cell typings at this point
    obsV = X.obs_vector("ind_cov")
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    ann_data_objects = [X[sgIndex == sgi, :] for sgi in range(len(sgUnique))]

    X = anndata.concat(ann_data_objects, axis=0)

    # keep only n observations
    X = X[::every_n, :]

    # get cell types -> only stay true if order of cells doesn't change after this
    cell_types = X.obs["cg_cov"].reset_index(drop=True)

    # get color mapping for patients by SLE status

    status = (
        X.obs[["ind_cov", "SLE_status"]]
        .sort_values(by="ind_cov")
        .drop_duplicates("ind_cov")
    )

    lut = {"SLE": "c", "Healthy": "m"}
    row_colors = status["SLE_status"].map(lut)

    assert np.all(np.isfinite(X.X.data))  # this should be true

    return tensorFy(X, third_axis), cell_types, row_colors
