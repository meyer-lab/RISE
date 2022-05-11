import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.config import config
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture
from jax import value_and_grad, grad

from scipy.optimize import minimize, Bounds
from tensorly.decomposition import partial_tucker, non_negative_parafac
from tensorly.cp_tensor import cp_normalize
from tensorly.tenalg import multi_mode_dot

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
config.update("jax_enable_x64", True)


def tensor_decomp(tensor: xa.DataArray, ranknumb: int):
    """Runs tensor decomposition on means tensor."""

    # Need to input the tMeans as numpy tensor
    fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)  # Normalizing factors

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))
        # For each dimension in tensor, have a specific ranking for each parameter

    return dfs, fac


def tensorcovar_decomp(tCovar: xa.DataArray, ranknumb: int):
    """Runs partial tucker decomposition on covariance tensor"""
    ptCore, ptFactors = partial_tucker(tCovar.to_numpy(), modes=[0, 3, 4, 5], rank=[ranknumb] * 4)

    return ptFactors, ptCore


def tensor_R2X(tensor: xa.DataArray, maxrank: int):
    """Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank + 1)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i])
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
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)

    factors = [jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    # Rebuidling factors and ranks

    factors_pt = [factors[0], factors[2], factors[3], factors[4]]
    ptNewCore = vectorIn[nN[-1] : :].reshape(rank, shape[1], shape[1], rank, rank, rank)

    return tl.cp_tensor.CPTensor((None, factors)), factors_pt, ptNewCore


def vector_guess(zflowTensor: xa.DataArray, rank: int, n_cluster: int):
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = np.sum(zflowTensor.shape) * rank

    factortotal = (
        factortotal
        - (len(zflowTensor.coords["Cell"]) * rank)
        + (rank * n_cluster)
        + ((rank**4) * (len(markerslist) ** 2))
        + n_cluster
    )

    return np.random.lognormal(mean=-1.0, size=factortotal)


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


def maxloglik_ptnnp(facVector, shape, rank:int, zflowTensor: xa.DataArray):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    rebuildnk = facVector[0 : shape[0]]

    factorsguess, rebuildPtFactors, rebuildPtCore = vector_to_cp_pt(facVector[shape[0] : :], rank, shape)
    rebuildMeans = tl.cp_to_tensor(factorsguess)

    rebuildPrecision = multi_mode_dot(rebuildPtCore, rebuildPtFactors, modes=[0, 3, 4, 5], transpose=False)
    rebuildPrecision = (rebuildPrecision + np.swapaxes(rebuildPrecision, 1, 2)) / 2.0  # Enforce symmetry
    # Creating function that we want to minimize
    return -comparingGMMjax(zflowTensor.to_numpy(), rebuildMeans, rebuildPrecision, rebuildnk)


def minimize_func(zflowTensor: xa.DataArray, rank: int, n_cluster: int):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    x0 = vector_guess(zflowTensor, rank, n_cluster)

    times = zflowTensor.coords["Time"]
    doses = zflowTensor.coords["Dose"]
    ligand = zflowTensor.coords["Ligand"]

    clustArray = np.arange(1, n_cluster + 1)
    meanShape = (n_cluster, len(markerslist), len(times), len(doses), len(ligand))
    commonDims = {"Time": times, "Dose": doses, "Ligand": ligand}
    coords={"Cluster": clustArray, "Markers": markerslist, **commonDims}

    args = (meanShape, rank, zflowTensor)

    tl.set_backend("jax")

    func = value_and_grad(maxloglik_ptnnp)

    def hvp(x, v, *args):
        return grad(lambda x: jnp.vdot(grad(maxloglik_ptnnp)(x, *args), v))(x)

    bnds = Bounds(np.zeros_like(x0), np.full_like(x0, np.inf), keep_feasible=True)
    opt = minimize(func, x0, bounds=bnds, jac=True, hessp=hvp, method="trust-constr", args=args, options={"verbose": 2, "maxiter": 50})

    tl.set_backend("numpy")

    rebuildCpFactors, _, ptNewCore = vector_to_cp_pt(opt.x[n_cluster : :], rank, meanShape)
    maximizedCpInfo = cp_normalize(rebuildCpFactors)
    maximizedNK = opt.x[0 : n_cluster]

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]

    maximizedFactors = []
    for ii, key in enumerate(coords):
        maximizedFactors.append(pd.DataFrame(maximizedCpInfo.factors[ii], columns=cmpCol, index=coords[key]))

    return maximizedNK, maximizedFactors, ptNewCore
