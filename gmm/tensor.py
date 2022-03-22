import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def get_conditions(df, X):
    """ Provides all unique conditions for a specific time, ligand, and concentration. """
    assert df.shape[0] % X.shape[0] == 0
    colNames = ["Time", "Ligand", "Dose"]
    cellsPerCondition = int(df.shape[0] / X.shape[0])
    conditions = df[colNames].iloc[::cellsPerCondition]
    conditions = conditions.set_index(colNames)
    return conditions


def tensor_means(meansDF: pd.DataFrame, means: np.ndarray):
    """ Turn the GMM mean results into a tensor form. """
    # Add valency to ligand name
    meansDF["Ligand"] = meansDF["Ligand"] + "-" + meansDF["Valency"].astype(str)

    conditions = get_conditions(meansDF, means)
    print(conditions)

    xd = xa.DataArray(means, dims=["Conditions", "Cluster", "Marker"])
    xd = xd.assign_coords(Cluster=np.arange(1, means.shape[1] + 1))
    xd = xd.assign_coords(Marker=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_covar(meansDF: pd.DataFrame, covar: np.ndarray):
    """ Turn the GMM covariance results into tensor form. """
    # Add valency to ligand name
    meansDF["Ligand"] = meansDF["Ligand"] + "-" + meansDF["Valency"].astype(str)

    conditions = get_conditions(meansDF, covar)

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

    return dfs,fac


def tensor_R2X(tensor, maxrank,tensortype):
    """ Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1,maxrank)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _,facinfo = tensor_decomp(tensor,rank[i],tensortype)
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank,varexpl

