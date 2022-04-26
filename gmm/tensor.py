import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.config import config
import tensorly as tl
import xarray as xa
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
config.update("jax_enable_x64", True)


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(
            np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(
            np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))

    return dfs, fac


def tensor_R2X(tensor: xa.DataArray, maxrank: int, tensortype):
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


def cp_to_vector(facinfo: tl.cp_tensor.CPTensor):
    """ Converts from factors to a linear vector. """
    vec = []

    for fac in facinfo.factors:
        vec = np.append(vec, fac.flatten())

    return vec


def vector_to_cp(vectorIn, rank: int, shape: tuple):
    """Converts linear vector to factors"""
    nN = jnp.cumsum(np.array(shape) * rank)
    nN = jnp.insert(nN, 0, 0)

    factors = [jnp.reshape(vectorIn[nN[ii]:nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    return tl.cp_tensor.CPTensor((None, factors))


def comparingGMM(zflowDF: xa.DataArray, tMeans: np.ndarray, tPrecision: np.ndarray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    X = zflowDF.to_numpy()

    it = np.nditer(tMeans[0, 0, :, :, :], flags=['multi_index', 'refs_ok'])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        Xcur = np.transpose(X[:, :, i, j, k])

        if np.all(np.isnan(Xcur)):  # Skip if there's no data
            continue

        gmm = GaussianMixture(n_components=nk.size, covariance_type="full", means_init=tMeans[:, :, i, j, k],
                              weights_init=nk)
        gmm._initialize(Xcur, np.ones((X.shape[1], nk.size)))
        gmm.precisions_cholesky_ = tPrecision[:, :, :, i, j, k]
        loglik += np.sum(gmm.score_samples(Xcur))

    return loglik


def comparingGMMjax(X, tMeans, tPrecision, nk):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nkl = jnp.log(nk / jnp.sum(nk))

    mp = jnp.einsum("ijklm,ijoklm->ioklm", tMeans, tPrecision)
    Xp = jnp.einsum("jiklm,njoklm->inoklm", X, tPrecision)
    log_prob = jnp.sum(jnp.square(Xp - mp[jnp.newaxis, :, :, :, :, :]), axis=2)
    log_prob = -0.5 * (X.shape[0] * jnp.log(2 * jnp.pi) + log_prob)

    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    ppp = tPrecision.reshape(tMeans.shape[0], -1, tPrecision.shape[3], tPrecision.shape[4], tPrecision.shape[5])
    log_det = jnp.sum(jnp.log(ppp[:, :: X.shape[0] + 1, :, :, :]), 1)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    loglik = jnp.sum(jsp.logsumexp(log_prob + log_det[jnp.newaxis, :, :, :, :] + nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis], axis=1))
    return loglik


def leastsquaresguess(nk, tMeans):
    nkCommon = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))  # nk is shared across conditions
    tMeans_vector = tMeans.values.flatten()
    return np.append(nkCommon, tMeans_vector)


def maxloglik(facVector, facInfo: tl.cp_tensor.CPTensor, tPrecision: xa.DataArray, nk: np.ndarray, zflowTensor: xa.DataArray):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    factorsguess = vector_to_cp(facVector, facInfo.rank, facInfo.shape)
    rebuildMeans = tl.cp_to_tensor(factorsguess)
    return -comparingGMMjax(zflowTensor.to_numpy(), rebuildMeans, tPrecision.to_numpy(), nk)
