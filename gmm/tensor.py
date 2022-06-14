from time import time_ns
import numpy as np
from sklearn.cluster import MeanShift
import jax.numpy as jnp
import jax.scipy.special as jsp
import tensorly as tl
from tqdm import tqdm
import xarray as xa
from copy import copy
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from jax import value_and_grad, jit, grad
from jax.experimental.host_callback import id_print

from scipy.optimize import minimize
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def vector_to_cp_pt(vectorIn, rank: int, shape: tuple):
    """Converts linear vector to factors"""
    vectorIn = jnp.exp(vectorIn)
    rebuildnk = vectorIn[0 : shape[0]]
    vectorIn = vectorIn[shape[0] : :]

    # Shape of tensor for means or precision matrix
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)

    factors = [jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    # Rebuidling factors and ranks

    precSym = jnp.zeros((shape[1], shape[1], rank))
    ai, bi = jnp.tril_indices(shape[1])
    pVec = vectorIn[nN[-1] : :].reshape(-1, rank)
    precSym = precSym.at[ai, bi, :].set(pVec)
    factors_pt = [factors[0], precSym, factors[2], factors[3], factors[4]]
    return rebuildnk, factors, factors_pt


def vector_guess(shape: tuple, rank: int):
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = np.sum(shape) * rank + int(shape[1] * (shape[1] - 1) / 2 + shape[1]) * rank + shape[0]
    vector = np.random.normal(loc=-1.0, size=factortotal)
    vector[0: shape[0]] = 1
    return vector


def comparingGMM(zflowDF: xa.DataArray, meanFact, tPrecision: np.ndarray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    tMeans = tl.cp_to_tensor((None, meanFact))
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


def comparingGMMjax(X, nk, meanFact: list, tPrecision):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    n_markers = tPrecision.shape[1]
    nkl = jnp.log(nk / jnp.sum(nk))
    mp = jnp.einsum("iz,jz,kz,lz,mz,ijoklm->ioklm", *meanFact, tPrecision)
    Xp = jnp.einsum("jiklm,njoklm->inoklm", X, tPrecision)
    log_prob = jnp.square(jnp.linalg.norm(Xp - mp[jnp.newaxis, :, :, :, :, :], axis=2))
    log_prob = -0.5 * (n_markers * jnp.log(2 * jnp.pi) + log_prob)

    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    ppp = jnp.diagonal(tPrecision, axis1=1, axis2=2)
    log_det = jnp.sum(jnp.log(ppp), axis=-1)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    loglik = jnp.sum(jsp.logsumexp(log_prob + log_det[jnp.newaxis, :, :, :, :] + nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis], axis=1))
    return loglik


def covFactor_to_precisions(covFac):
    """Convert from the cholesky decomposition of the covariance matrix, to the precision matrix."""
    cholBuilt = jnp.einsum("ax,bcx,dx,ex,fx->abcdef", *covFac)
    cholMv = jnp.moveaxis(cholBuilt, (1, 2), (4, 5))
    cholMv = jnp.linalg.inv(cholMv)
    precBuild = jnp.moveaxis(cholMv, (4, 5), (1, 2))
    assert cholBuilt.shape == precBuild.shape
    return jnp.swapaxes(precBuild, 1, 2)


def maxloglik_ptnnp(facVector, shape: tuple, rank: int, X):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    nk, meanFact, covFac = vector_to_cp_pt(facVector, rank, shape)
    precBuild = covFactor_to_precisions(covFac)

    # Creating function that we want to minimize
    return -comparingGMMjax(X, nk, meanFact, precBuild)


def minimize_func(zflowTensor: xa.DataArray, rank: int, n_cluster: int, maxiter=300, x0=None):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    meanShape = (n_cluster, zflowTensor.shape[0], zflowTensor.shape[2], zflowTensor.shape[3], zflowTensor.shape[4])

    args = (meanShape, rank, zflowTensor.to_numpy())
    func = jit(value_and_grad(maxloglik_ptnnp), static_argnums=(1, 2))

    if x0 is None:
        x0 = vector_guess(meanShape, rank)

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp, static_argnums=(2, 3))

    tq = tqdm(total=maxiter, delay=0.1)

    def callback(xk, state):
        gNorm = np.linalg.norm(state.grad)
        tq.set_postfix(val='{:.2e}'.format(state.fun), g='{:.2e}'.format(gNorm), refresh=False)
        tq.update(1)

    opts = {"maxiter": maxiter, "disp": False}
    bounds = ((np.log(1e-1), np.log(1e1)), ) * n_cluster + ((np.log(1e-4), np.log(50.0)), ) * (len(x0) - n_cluster)
    opt = minimize(func, x0, jac=True, hessp=hvpj, callback=callback, method="trust-constr", bounds=bounds, args=args, options=opts)
    tq.close()

    optNK, optCP, optPT = vector_to_cp_pt(opt.x, rank, meanShape)
    optLL = -opt.fun
    preNormCP = copy(optCP)
    optCP = cp_normalize((None, optCP))
    optVec = opt.x

    return optNK, optCP, optPT, optLL, optVec, preNormCP


def tensorGMM_CV(X, numFolds: int, numClusters: int, numRank: int, maxiter=300):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0.0
    meanShape = (numClusters, X.shape[0], X.shape[2], X.shape[3], X.shape[4])

    kf = KFold(n_splits=numFolds)
    x0 = None

    # Start generating splits and running model
    for train_index, test_index in kf.split(X[:, :, 0, 0, 0].T):
        # Train
        _, _, _, _, x0, _ = minimize_func(X[:, train_index, :, :, :], numRank, numClusters, maxiter=maxiter, x0=x0)
        # Test
        test_ll = -maxloglik_ptnnp(x0, meanShape, numRank, X[:, test_index, :, :, :].to_numpy())
        logLik += test_ll

    return float(logLik)


def gen_points_GMM(optNK, optCP, optPT, time):
    """Generates points from a scikit-learn GMM object for a fit NK, CP and PT"""
    GMM = GaussianMixture(n_components=optNK.size, covariance_type="full").fit([[120, 0], [2, 1], [4, 2], [6, 3], [0, 4], [0, 5], [0, 5], [24, 5], [81, 1], [98, 3], [12, 8], [87, 5], [3, 5]])
    precisions = covFactor_to_precisions(optPT)
    means = jnp.einsum("iz,jz,kz,lz,mz,ijoklm->ioklm", *optCP, precisions)
    precisions = precisions[:, :, :, time, 0, 0].reshape([6, 2, 2])
    covariances = jnp.linalg.inv(precisions)

    for i in range(0, covariances.shape[0]):
        covariances = covariances.at[i, 1, 0].set(covariances[i, 0, 1])
        precisions = precisions.at[i, 1, 0].set(precisions[i, 0, 1])
    GMM.precisions_ = np.array(precisions)
    GMM.precisions_cholesky_ = np.array(np.linalg.cholesky(precisions))
    
    
    GMM.weights_ = np.array(optNK / np.sum(optNK))
    GMM.means_ = np.array(means[:, :, time, 0, 0].reshape([6, 2]))
    GMM.covariances_ = np.array(covariances)
    GMM.converged_ = True
    points = GMM.sample(500)
    return points[0]
