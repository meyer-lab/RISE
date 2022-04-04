import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def tensor_means(conditions, means: np.ndarray):
    """ Turn the GMM mean results into a tensor form. """
    xd = xa.DataArray(means, dims=["Conditions", "Cluster", "Marker"])
    xd = xd.assign_coords(Cluster=np.arange(1, means.shape[1] + 1))
    xd = xd.assign_coords(Marker=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_covar(conditions, covar: np.ndarray):
    """ Turn the GMM covariance results into tensor form. """
    # covar is conditions x clusters x markers x markers
    xd = xa.DataArray(covar, dims=["Conditions", "Cluster", "Marker1", "Marker2"])
    xd = xd.assign_coords(Cluster=np.arange(1, covar.shape[1] + 1))
    xd = xd.assign_coords(Marker1=markerslist, Marker2=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))

    return dfs, fac


def tensor_R2X(tensor, maxrank, tensortype):
    """ Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i], tensortype)
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank, varexpl


def meanCP_to_DF(factorinfo_NNP, tMeans):
    """Converts output of factor decomposition into a dataframe"""
    newTens = tl.cp_to_tensor(factorinfo_NNP)
    return xa.DataArray(newTens, dims=tMeans.dims, coords=tMeans.coords)
