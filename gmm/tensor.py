import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
import tensorly as tl
from tqdm import tqdm
import xarray as xa
from copy import deepcopy
from sklearn.model_selection import KFold
from jax import value_and_grad, jit, grad
from jax.lax.linalg import triangular_solve
from scipy.optimize import minimize, Bounds
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def vector_to_cp_pt(vectorIn, rank: int, shape: tuple, nk_rearrange=False):
    """Converts linear vector to factors"""
    vectorIn = jnp.exp(vectorIn)
    if nk_rearrange is False:
        rebuildnk = vectorIn[0 : shape[0]]
        vectorIn = vectorIn[shape[0] : :]

    else:
        rebuildnk = vectorIn[0 : (shape[0] * rank)]
        rebuildnk = jnp.reshape(rebuildnk, (shape[0], rank))
        vectorIn = vectorIn[(shape[0] * rank) : :]

    # Shape of tensor for means or precision matrix
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)

    factors = [
        jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank))
        for ii in range(len(shape))
    ]
    # Rebuidling factors and ranks

    precisions = jnp.zeros((shape[1], shape[1], rank))
    ai, bi = jnp.tril_indices(shape[1])
    pVec = vectorIn[nN[-1] : :].reshape(-1, rank)
    precisions = precisions.at[ai, bi, :].set(pVec)
    factors_pt = [factors[0], precisions, factors[2], factors[3], factors[4]]
    return rebuildnk, factors, factors_pt


def vector_guess(shape: tuple, rank: int, seed=None, nk_rearrange=False):
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = (
        np.sum(shape) * rank
        + int(shape[1] * (shape[1] - 1) / 2 + shape[1]) * rank
        + shape[0]
    )

    if (nk_rearrange is True) and (rank > 1):
        factortotal += (rank - 1) * shape[0]

    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=-1.0, size=factortotal)
    vector[0 : shape[0]] = np.log(1.0 / shape[0])

    return vector


def log_prob_and_det(X, mp, tPrecision):
    n_markers = tPrecision.shape[1]
    Xp = jnp.einsum("ji...,njo...->ino...", X, tPrecision)
    log_prob = jnp.square(jnp.linalg.norm(Xp - mp[jnp.newaxis, ...], axis=2))
    log_prob = -0.5 * (n_markers * jnp.log(2 * jnp.pi) + log_prob)

    # Need to check here for the sum
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    ppp = jnp.diagonal(tPrecision, axis1=1, axis2=2)
    log_det = jnp.sum(jnp.log(ppp), axis=-1)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return log_prob + log_det[jnp.newaxis, ...]


def comparingGMMjax(X, nk, meanFact: list, tPrecision):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nkl = jnp.log(nk / jnp.sum(nk))
    mp = jnp.einsum("iz,jz,kz,lz,mz,ijoklm->ioklm", *meanFact, tPrecision)

    log_prob_det = log_prob_and_det(X, mp, tPrecision)
    loglik = jnp.sum(
        jsp.logsumexp(
            log_prob_det + nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis],
            axis=1,
        )
    )
    return loglik


def comparingGMMjax_NK(X, nkFact, meanFact: list, tPrecision):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nkFact.ndim == 2
    nk = nkFact @ meanFact[2].T
    assert nk.ndim == 2
    nkl = jnp.log(nk / jnp.sum(nk, axis=0, keepdims=True))
    mp = jnp.einsum("iz,jz,kz,ijok->iok", *meanFact, tPrecision)

    log_prob_det = log_prob_and_det(X, mp, tPrecision)
    loglik = jnp.sum(jsp.logsumexp(log_prob_det + nkl[jnp.newaxis, :, :], axis=1))
    return loglik


def covFactor_to_precisions(covFac, returnCov=False):
    """Convert from the cholesky decomposition of the covariance matrix, to the precision matrix."""
    cov_chol = jnp.einsum("ax,bcx,dx,ex,fx->abcdef", *covFac)
    origShape = cov_chol.shape
    if returnCov:
        return cov_chol
    cov_chol = jnp.moveaxis(cov_chol, (1, 2), (4, 5))
    Y = jnp.broadcast_to(
        jnp.eye(cov_chol.shape[4])[
            jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :
        ],
        cov_chol.shape,
    )
    assert cov_chol.shape == Y.shape
    prec_chol = triangular_solve(cov_chol, Y, lower=True)
    prec_chol = jnp.moveaxis(prec_chol, (4, 5), (1, 2))
    assert origShape == prec_chol.shape
    return prec_chol


def maxloglik_ptnnp(facVector, shape: tuple, rank: int, X):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    nk, meanFact, covFac = vector_to_cp_pt(facVector, rank, shape)
    prec_chol = covFactor_to_precisions(covFac)

    # Creating function that we want to minimize
    return -comparingGMMjax(X, nk, meanFact, prec_chol) / X.shape[1]


def maxloglik_ptnnp_NK(facVector, shape: tuple, rank: int, X):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    nk, meanFact, covFac = vector_to_cp_pt(facVector, rank, shape, nk_rearrange=True)
    prec_chol = covFactor_to_precisions(covFac)

    meanFact = [meanFact[0], meanFact[1], meanFact[2]]

    prec_chol = jnp.reshape(prec_chol, prec_chol.shape[0:4])
    X = jnp.reshape(X, X.shape[0:3])

    # Creating function that we want to minimize
    return -comparingGMMjax_NK(X, nk, meanFact, prec_chol) / X.shape[1]


def minimize_func(
    X: xa.DataArray,
    rank: int,
    n_cluster: int,
    maxiter=400,
    verbose=True,
    x0=None,
    nk_rearrange=False,
    seed=None,
):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    meanShape = (n_cluster, X.shape[0], X.shape[2], X.shape[3], X.shape[4])

    args = (meanShape, rank, X.to_numpy())

    if nk_rearrange is False:
        func = jit(value_and_grad(maxloglik_ptnnp), static_argnums=(1, 2))
    else:
        func = jit(value_and_grad(maxloglik_ptnnp_NK), static_argnums=(1, 2))

    if x0 is None:
        x0 = vector_guess(meanShape, rank, seed=seed, nk_rearrange=nk_rearrange)

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp, static_argnums=(2, 3))

    tq = tqdm(total=maxiter, delay=0.1, disable=(verbose is False))

    def callback(_, state):
        gNorm = np.linalg.norm(state.grad)
        tq.set_postfix(
            val="{:.2e}".format(state.fun), g="{:.2e}".format(gNorm), refresh=False
        )
        tq.update(1)

    opts = {"maxiter": maxiter, "disp": False}

    # Add bounds
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[0:n_cluster] = np.log(0.1)
    ub[0:n_cluster] = 0.0
    bounds = Bounds(lb, ub, keep_feasible=True)

    opt = minimize(
        func,
        x0,
        jac=True,
        hessp=hvpj,
        callback=callback,
        method="trust-constr",
        bounds=bounds,
        args=args,
        options=opts,
    )
    tq.close()

    optNK, optCP, optPT = vector_to_cp_pt(
        np.copy(opt.x), rank, meanShape, nk_rearrange=nk_rearrange
    )

    preNormCP = deepcopy(optCP)

    return optNK, cp_normalize((None, optCP)), optPT, -opt.fun, opt.x, preNormCP


def tensorGMM_CV(X, numFolds: int, numClusters: int, numRank: int, maxiter=200):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0.0
    meanShape = (numClusters, X.shape[0], X.shape[2], X.shape[3], X.shape[4])

    kf = KFold(n_splits=numFolds)
    x0 = None

    # Start generating splits and running model
    for train_index, test_index in kf.split(X[:, :, 0, 0, 0].T):
        # Train
        _, _, _, _, x0, _ = minimize_func(
            X[:, train_index, :, :, :],
            numRank,
            numClusters,
            maxiter=maxiter,
            verbose=False,
            x0=x0,
        )
        # Test
        test_ll = -maxloglik_ptnnp(
            x0, meanShape, numRank, X[:, test_index, :, :, :].to_numpy()
        )
        logLik += test_ll

    return float(logLik)


def sample_GMM(weights_, means_, cholCovs, n_samples):
    n_samples_comp = np.random.multinomial(n_samples, weights_)

    X = np.vstack(
        [
            np.random.multivariate_normal(mean, cholCov @ cholCov.T, int(sample))
            for (mean, cholCov, sample) in zip(means_, cholCovs, n_samples_comp)
        ]
    )
    y = np.concatenate(
        [np.full(sample, j + 1, dtype=int) for j, sample in enumerate(n_samples_comp)]
    )
    return X, y


def gen_points_GMM(optNK, optCP, optPT, time, dose, ligand, n_samples=1000):
    """Generates points from a scikit-learn GMM object for a fit NK, CP and PT"""
    cholCov = covFactor_to_precisions(optPT, returnCov=True)
    cholCov = np.squeeze(cholCov[:, :, :, time, dose, ligand])
    means = tl.cp_to_tensor((None, optCP))

    nk = optNK / np.sum(optNK)
    means = np.squeeze(means[:, :, time, dose, ligand])
    samples = sample_GMM(nk, means, cholCov, n_samples)
    return samples


def optimal_seed(n_seeds, *args, **kwargs):
    """Finds the optimal seed number to minimize log likeihood"""
    total_loglik = []
    for i in range(n_seeds):
        _, _, _, x, _, _ = minimize_func(*args, **kwargs, seed=i)
        total_loglik = np.append(total_loglik, x)

    optimalseed = np.argmin(total_loglik)
    min_loglik = np.min(total_loglik)

    return optimalseed, min_loglik
