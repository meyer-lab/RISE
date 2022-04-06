import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD4", "CD45RA", "pSTAT5"]


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(
            np.nan_to_num(
                tensor.to_numpy()), mask=np.isfinite(
                tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(
            np.nan_to_num(
                tensor.to_numpy()), mask=np.isfinite(
                tensor.to_numpy()), rank=ranknumb)

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


def comparingGMM(zflowDF: pd.DataFrame, tMeans: xa.DataArray, tCovar: xa.DataArray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    conditions = zflowDF.groupby(["Ligand", "Dose", "Time"])
    loglik = 0.0

    for name, cond_cells in conditions:
        # Means of GMM
        flow_mean = tMeans.loc[:, markerslist, name[0], name[1], name[2]]
        flow_covar = tCovar.loc[:, markerslist, markerslist, name[0], name[1], name[2]]
        assert flow_mean.shape[0] == flow_covar.shape[0]  # Rows are clusters
        assert flow_mean.size > 0
        assert flow_covar.size > 0
        assert np.all(np.isfinite(flow_mean))
        assert np.all(np.isfinite(flow_covar))

        X = cond_cells[markerslist].to_numpy()

        gmm = GaussianMixture(
            n_components=nk.size,
            covariance_type="full",
            means_init=flow_mean.to_numpy(),
            weights_init=nk)
        gmm._initialize(X, np.ones((X.shape[0], nk.size)))
        gmm.precisions_cholesky_ = _compute_precision_cholesky(flow_covar, "full")

        loglik += np.sum(gmm.score_samples(X))

    return loglik
