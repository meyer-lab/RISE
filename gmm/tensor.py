import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.config import config
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture

from tensorly.decomposition import non_negative_parafac, parafac, partial_tucker
from tensorly.cp_tensor import cp_normalize
from tensorly.tenalg import multi_mode_dot

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
config.update("jax_enable_x64", True)


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """Runs tensor decomposition on means tensor."""

    # Need to input the tMeans as numpy tensor
    if tensortype == "NNparafac":
        fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)  # Normalizing factors

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))
        # For each dimension in tensor, have a specific ranking for each parameter

    return dfs, fac


def tensorcovar_decomp(tCovar: xa.DataArray, ranknumb: int):
    """Runs partial tucker decomposition on covariance tensor"""
    ptCore, ptFactors = partial_tucker(tCovar.to_numpy(), modes=[0, 3, 4, 5], rank=ranknumb)

    return ptFactors, ptCore


def tensor_R2X(tensor: xa.DataArray, maxrank: int, tensortype):
    """Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank + 1)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i], tensortype)
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        # Need to rebuild tensor using factors and weights
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank, varexpl


def cp_pt_to_vector(facinfo: tl.cp_tensor.CPTensor, ptCore):
    """Converts from factors to a linear vector."""
    vec = np.array([], dtype=float)

    for fac in facinfo.factors:
        vec = np.append(vec, fac.flatten())

    vec = np.append(vec, ptCore.flatten())

    return vec


def vector_to_cp_pt(vectorIn, rank: int, shape: tuple):
    """Converts linear vector to factors"""
    # Shape of tensor for means or precision matrix
    nN = jnp.cumsum(np.array(shape) * rank)
    nN = jnp.insert(nN, 0, 0)

    factors = [jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    # Rebuidling factors and ranks

    factors_pt = [factors[0], factors[2], factors[3], factors[4]]
    ptNewCore = vectorIn[nN[-1]::].reshape(rank, shape[1], shape[1], rank, rank, rank)

    return tl.cp_tensor.CPTensor((None, factors)), factors_pt, ptNewCore


def comparingGMM(zflowDF: xa.DataArray, tMeans: np.ndarray, tPrecision: np.ndarray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    X = zflowDF.to_numpy()

    it = np.nditer(tMeans[0, 0, :, :, :], flags=["multi_index", "refs_ok"])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        Xcur = np.transpose(X[:, :, i, j, k])  # Cell Number per experiment x Marker

        if np.all(np.isnan(Xcur)):  # Skip if there's no data
            continue

        gmm = GaussianMixture(n_components=nk.size, covariance_type="full", means_init=tMeans[:, :, i, j, k], weights_init=nk)
        gmm._initialize(Xcur, np.ones((X.shape[1], nk.size)))  # Markers x Clusters
        gmm.precisions_cholesky_ = tPrecision[:, :, :, i, j, k]  # Cluster x Marker x Marker
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


def maxloglik_ptnnp(facVector, facInfo: tl.cp_tensor.CPTensor, zflowTensor: xa.DataArray):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    rebuildnk = facVector[0 : facInfo.shape[0]]

    factorsguess, rebuildPtFactors, rebuildPtCore = vector_to_cp_pt(facVector[facInfo.shape[0]::], facInfo.rank, facInfo.shape)
    rebuildMeans = tl.cp_to_tensor(factorsguess)

    rebuildPrecision = multi_mode_dot(rebuildPtCore, rebuildPtFactors, modes=[0, 3, 4, 5], transpose=False)
    rebuildPrecision = jnp.abs(rebuildPrecision)  # TODO: Remove this eventually.
    rebuildPrecision = (rebuildPrecision + np.swapaxes(rebuildPrecision, 1, 2)) / 2.0  # Enforce symmetry
    # Creating function that we want to minimize
    return -comparingGMMjax(zflowTensor.to_numpy(), rebuildMeans, rebuildPrecision, rebuildnk)
