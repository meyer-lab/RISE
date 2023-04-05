from os.path import dirname
import numpy as np
import pandas as pd
from scipy.stats import linregress
import anndata
from ..parafac2 import Pf2X


path_here = dirname(dirname(__file__))


def xarrayIfy(annD, obsV):
    sgUnique, sgIndex, sgCounts = np.unique(
        obsV, return_inverse=True, return_counts=True
    )

    data = list()

    for sgi in range(len(sgUnique)):
        x_temp = annD[sgIndex == sgi, :]
        assert x_temp.shape[0] == sgCounts[sgi]
        data.append(x_temp.X.toarray())

    return Pf2X(data, sgUnique, annD.var_names)


def import_perturb_RPE():
    ds_disk = anndata.read_h5ad("/opt/andrew/rpe1_normalized_singlecell_01.h5ad")

    # Remove NaNs
    ds_disk = ds_disk[:, np.all(np.isfinite(ds_disk.X), axis=0)]

    sgRNAs = ds_disk.obs_vector("sgID_AB")
    return xarrayIfy(ds_disk, sgRNAs)


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper."""

    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/meta.csv")

    # Cell barcodes (33482)
    barcodes = pd.read_csv("sccp/data/barcodes.tsv", sep="\t", header=None, names=("cell_barcode", ))

    # Left merging should put the barcodes in order
    metafile = pd.merge(barcodes, metafile, on="cell_barcode", how='left', validate="one_to_one")

    # h5ad is simplified version of mtx format
    # data = sc.read_10x_mtx("/opt/andrew/Thompson", var_names='gene_symbols', make_unique=True)
    # data.write('thompson.h5ad', compression="gzip")
    data = anndata.read_h5ad("/opt/andrew/thompson.h5ad", chunk_size=12000)

    data.obs["Drugs"] = pd.Categorical(metafile["sample_id"])
    return data


def mu_sigma_normalize(X: anndata.AnnData, scalingfactor: float):
    """Calculates the mu and sigma for every gene and returns
    means, sigmas, and dataframe filtered for genes expressed
    in > 0.1% of cells."""
    assert np.all(np.isfinite(X.X.data))
    X.X = X.X.todense()

    X = X[:, np.mean(X.X > 0, axis=0) > 0.001]
    X.X /= np.sum(X.X, axis=0)

    # Only operating on the data works because 0 ends up as 0 here
    X.X = np.log10((scalingfactor * X.X) + 1)
    means = np.mean(X.X, axis=0)
    cv = np.std(X.X, axis=0) / means

    return X, np.log10(means + 1e-10), np.log10(cv + 1e-10)


def ThompsonXA_SCGenes(offset=1.0):
    """Turns filtered and normalized cells into an Xarray."""
    genesDF = import_thompson_drug()
    X, logmean, logstd = mu_sigma_normalize(genesDF, scalingfactor=1000)

    if offset != 1.0:
        slope, intercept, _, _, _ = linregress(logmean, logstd)

        above_idx = logstd > logmean * slope + intercept + np.log10(offset)
        X = X[:, above_idx]

    # Assign cells a count per-experiment so we can reindex
    data = xarrayIfy(X, X.obs_vector("Drugs"))
    return data
