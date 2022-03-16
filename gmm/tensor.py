import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa

from tensorly.decomposition import non_negative_parafac, parafac


def tensor_means(meansDF: pd.DataFrame, markerslist: list):
    """ Turn the GMM mean results into a tensor form. """
    # Add valency to ligand name
    meansDF["Ligand"] = meansDF["Ligand"] + "-" + meansDF["Valency"].astype(str)

    tensor = meansDF.set_index(["Time", "Ligand", "Dose", "Cluster"]).to_xarray()
    return tensor[markerslist].to_array(dim="Marker")


def tensor_covar(meansDF: pd.DataFrame, markerslist: list, covar: np.ndarray):
    """ Turn the GMM covariance results into tensor form. """
    colNames = ["Time", "Ligand", "Dose"]
    conditions = meansDF[colNames].drop_duplicates().set_index(colNames)

    # covar is conditions x clusters x markers x markers
    xd = xa.DataArray(covar, dims=["Conditions", "Cluster", "Marker1", "Marker2"])
    xd = xd.assign_coords(Cluster=np.arange(1, covar.shape[1] + 1))
    xd = xd.assign_coords(Marker1=markerslist, Marker2=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ X """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))

    return dfs
