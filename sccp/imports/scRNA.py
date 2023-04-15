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
    # data = sc.read_10x_mtx("/opt/andrew/Thompson", var_names='gene_symbols', make_unique=True)
    # data.write('thompson.h5ad', compression="gzip")
    data = anndata.read_h5ad("/opt/andrew/thompson.h5ad", chunk_size=12000)

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
