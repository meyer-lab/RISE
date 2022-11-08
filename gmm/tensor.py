import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
import tensorly as tl
from tqdm import tqdm
import xarray as xa
from sklearn.model_selection import KFold
from jax import value_and_grad, jit
from jax.lax.linalg import triangular_solve
from scipy.optimize import minimize, Bounds
from copy import copy


markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def infer_rank(length: int, shape: tuple[int, ...], nk_rearrange: bool = False):
    """Infer the rank from the vector length."""
    if nk_rearrange:
        return int(
            length
            / (np.sum(shape) + 2 * shape[0] + shape[2] + np.sum(shape[2::]) + int(shape[1] * (shape[1] - 1) / 2 + shape[1]))
        )

    length -= shape[0]
    return int(length / (np.sum(shape) + shape[0] + np.sum(shape[2::]) + int(shape[1] * (shape[1] - 1) / 2 + shape[1])))


def sample_GMM(
    weights: jnp.ndarray, means: jnp.ndarray, covars: jnp.ndarray, n_samples: int
):
    n_samples_comp = np.random.multinomial(n_samples, weights)

    X = np.vstack(
        [
            np.random.multivariate_normal(mean, covar, int(sample))
            for (mean, covar, sample) in zip(means, covars, n_samples_comp)
        ]
    )
    y = np.concatenate(
        [np.full(sample, j + 1, dtype=int) for j, sample in enumerate(n_samples_comp)]
    )
    return X.T, y


class tensorGMM(tl.cp_tensor.CPTensor):
    def __init__(
        self, vectorIn: jnp.ndarray, shape: tuple[int, ...], nk_rearrange: bool = False
    ):
        """Build factor matrices from vector and shapes."""
        rank = infer_rank(vectorIn.size, shape, nk_rearrange)
        self.nk_rearrange = nk_rearrange
        if nk_rearrange:
            self.nk = vectorIn[0: (shape[0] * rank)]
            self.nk = jnp.reshape(self.nk, (shape[0], rank))
            vectorIn = vectorIn[(shape[0] * rank)::]
            self.nk_Fac = jnp.exp(vectorIn[0: (shape[2] * rank)])
            self.nk_Fac = jnp.reshape(self.nk_Fac, (shape[2], rank))
            vectorIn = vectorIn[(shape[2] * rank)::]
        else:
            self.nk = vectorIn[0: shape[0]]
            vectorIn = vectorIn[shape[0]::]

        # Shape of tensor for means or precision matrix
        nN = np.cumsum(np.array(shape) * rank)
        nN = np.insert(nN, 0, 0)

        super().__init__(
            (
                None,
                [
                    jnp.reshape(vectorIn[nN[ii]: nN[ii + 1]], (shape[ii], rank))
                    for ii in range(len(shape))
                ],
            )
        )
        
        # Reshapes non-signal covariance factors from vector
        vectorIn = vectorIn[nN[-1]::]
        nN[2::] -= rank * shape[1]
        nN = np.delete(nN, 2)
        covarFacS = list(shape)
        covarFacS.pop(1)
        covFacs = []

        for ii in range(len(covarFacS)):
            covFacs.append(jnp.exp(jnp.reshape(vectorIn[nN[ii]: nN[ii + 1]], (covarFacS[ii], rank))))

        self.covFacs = covFacs
        covars = jnp.zeros((shape[1], shape[1], rank))
        ai, bi = jnp.tril_indices(shape[1])
        pVec = vectorIn[nN[-1]::].reshape(-1, rank)
        self.covars = covars.at[ai, bi, :].set(jnp.exp(pVec))

        self.covars, self.covFacs, self.covWeights = norm_covariances(self.covars, self.covFacs)

    def get_precisions(self) -> jnp.ndarray:
        """Return precision matrices."""
        cov_chol = self.get_covariances()
        cov_chol = jnp.moveaxis(cov_chol, (1, 2), (4, 5))  # move axes
        Y = jnp.broadcast_to(
            jnp.eye(cov_chol.shape[4])[None, None, None, None, :, :],
            cov_chol.shape,
        )
        prec_chol = triangular_solve(cov_chol, Y, lower=True)
        prec_chol = jnp.moveaxis(prec_chol, (4, 5), (1, 2))  # move axes back
        return prec_chol

    def get_covariances(self) -> jnp.ndarray:
        """Return covariance matrices."""
        return jnp.einsum(
            "ax,bcx,dx,ex,fx->abcdef",
            self.covFacs[0],
            self.covars * self.covWeights[jnp.newaxis, jnp.newaxis, :],
            *self.covFacs[1::],)

    def get_covariances_xarray(self, X):
        """Return covariance matrices."""
        covar = self.get_covariances()

        coordinates = {"Cluster": np.arange(1, self.shape[0] + 1),
                       "Signal1": X.coords[X.dims[0]],
                       "Signal2": X.coords[X.dims[0]],
                       X.dims[2]: X.coords[X.dims[2]],
                       X.dims[3]: X.coords[X.dims[3]],
                       X.dims[4]: X.coords[X.dims[4]]}

        covar_xarray = xa.DataArray(covar,
                                    coords={**coordinates})

        return covar_xarray

    def log_det_prec(self) -> jnp.ndarray:
        """Return the determinants of the precisions matrices."""
        # Need to check here for the sum
        # The determinant of the precision matrix from the Cholesky decomposition
        # corresponds to the negative half of the determinant of the full precision matrix.
        # In short: det(precision_chol) = - det(precision) / 2
        # The precision determinant is the negative of the covariance one.
        # Therefore, we'll calculate the covariance determinant which may be a little more accurate.
        tC = self.get_covariances()
        lC = jnp.sum(jnp.log(jnp.diagonal(tC, axis1=1, axis2=2)), axis=-1)
        return -lC

    def sample(self, n_samples: int = 1000):
        """Generate a random dataset from the factors."""
        cholCov = self.get_covariances()
        covars = np.einsum("ijk...,ikj...->ijk...", cholCov, cholCov)
        means = tl.cp_to_tensor(self)

        assert self.nk_rearrange is False
        nk = self.nk / np.sum(self.nk)

        X = np.empty((means.shape[1], n_samples, *means.shape[2::]))
        y = np.empty((n_samples, *means.shape[2::]))
        for i, j, k in np.ndindex(X.shape[2::]):
            X[:, :, i, j, k], y[:, i, j, k] = sample_GMM(
                nk, means[:, :, i, j, k], covars[:, :, :, i, j, k], n_samples
            )

        return X, y

    def get_factors_xarray(self, X):
        cp_factors = tl.cp_normalize(self)
        cmpCol = [f"Cmp. {i}" for i in np.arange(1, cp_factors.rank + 1)]
        coordinates = {"Cluster": np.arange(1, cp_factors.shape[0] + 1),
                       X.dims[0]: X.coords[X.dims[0]],
                       X.dims[2]: X.coords[X.dims[2]],
                       X.dims[3]: X.coords[X.dims[3]],
                       X.dims[4]: X.coords[X.dims[4]]}

        da = xa.Dataset({"Dimension1": (["Cluster", "Cmp"], cp_factors.factors[0]),
                         "Dimension2": ([X.dims[0], "Cmp"], cp_factors.factors[1]),
                         "Dimension3": ([X.dims[2], "Cmp"], cp_factors.factors[2]),
                         "Dimension4": ([X.dims[3], "Cmp"], cp_factors.factors[3]),
                         "Dimension5": ([X.dims[4], "Cmp"], cp_factors.factors[4])},
                        coords={"Cmp": cmpCol, **coordinates})

        return da
    
    
    def get_covariance_factors(self, X):
        """Outputs xarray of all covariance factors for each dimension"""
        cmpCol = [f"Cmp. {i}" for i in np.arange(1, self.rank + 1)]
        coordinates = {"Cluster": np.arange(1, self.shape[0] + 1),
                       X.dims[0]: X.coords[X.dims[0]],
                       X.dims[2]: X.coords[X.dims[2]],
                       X.dims[3]: X.coords[X.dims[3]],
                       X.dims[4]: X.coords[X.dims[4]]}
        
        cov_fac = xa.Dataset({"Dimension1": (["Cluster", "Cmp"], self.covFacs[0]),
                         "Dimension2": ([X.dims[0], X.dims[0], "Cmp"], self.covars),
                         "Dimension3": ([X.dims[2], "Cmp"], self.covFacs[1]),
                         "Dimension4": ([X.dims[3], "Cmp"], self.covFacs[2]),
                         "Dimension5": ([X.dims[4], "Cmp"], self.covFacs[3])},
                        coords={"Cmp": cmpCol, **coordinates})
        
        return cov_fac

    def norm_NK(self) -> jnp.ndarray:
        """Normalizes NK values to percentages"""
        return self.nk / np.sum(self.nk)


def norm_covariances(covars, covFacs, covWeights=None):
    """Normalizes covariance factors. Weight is moved to inside covFacs."""
    normalized_factors = []
    if covWeights is None:
        covWeights = jnp.ones(covFacs[0].shape[1])

    for factor in covFacs:
        scales = jnp.linalg.norm(factor, axis=0)
        normalized_factors.append(factor / scales[jnp.newaxis, :])
        covWeights *= scales

    scales = jnp.linalg.norm(covars, axis=(0, 1))
    covars /= scales[jnp.newaxis, jnp.newaxis, :]
    covWeights *= scales
    
    return covars, normalized_factors, covWeights


def vector_guess(
    shape: tuple[int], rank: int, seed=None, nk_rearrange: bool = False
) -> np.ndarray:
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = (
        np.sum(shape) * rank
        + shape[0] * rank
        + np.sum(shape[2::]) * rank
        + int(shape[1] * (shape[1] - 1) / 2 + shape[1]) * rank
        + shape[0]
    )

    if (nk_rearrange is True) and (rank > 1):
        factortotal += (rank - 1) * shape[0]
        factortotal += rank * shape[2]

    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0, size=factortotal)

    if nk_rearrange:
        vector[0: shape[0] * rank] = 1.0
    else:
        vector[0: shape[0]] = 1.0

    return vector


def comparingGMMjax(X: np.ndarray, facBuild: tensorGMM) -> jnp.ndarray:
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    tPrecision = facBuild.get_precisions()
    log_det = facBuild.log_det_prec()
    nk = facBuild.nk

    if facBuild.nk_rearrange:
        assert nk.ndim == 2
        nk = nk @ facBuild.nk_Fac.T
        assert nk.ndim == 2
        nkl = jnp.log(nk / jnp.sum(nk, axis=0, keepdims=True))
        nkl = nkl[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis]
    else:
        assert nk.ndim == 1
        nkl = jnp.log(nk / jnp.sum(nk))
        nkl = nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    mp = jnp.einsum("iz,jz,kz,lz,mz,ijoklm->ioklm", *facBuild.factors, tPrecision)

    n_markers = tPrecision.shape[1]
    Xp = jnp.einsum("ji...,njo...->ino...", X, tPrecision)
    log_prob = jnp.square(jnp.linalg.norm(Xp - mp[jnp.newaxis, ...], axis=2))
    log_prob = -0.5 * (n_markers * jnp.log(2 * jnp.pi) + log_prob)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return jnp.sum(jsp.logsumexp(log_prob + log_det[jnp.newaxis, ...] + nkl, axis=1))


def cell_assignment(X: np.ndarray, facBuild: tensorGMM) -> jnp.ndarray:
    """Provides the cell assignmens to each cluster, given the dataset and factors."""
    tPrecision = facBuild.get_precisions()
    nk = facBuild.nk

    if facBuild.nk_rearrange:
        assert nk.ndim == 2
        nk = nk @ facBuild.nk_Fac.T
        assert nk.ndim == 2
        nkl = jnp.log(nk / jnp.sum(nk, axis=0, keepdims=True))
        nkl = nkl[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis]
    else:
        assert nk.ndim == 1
        nkl = jnp.log(nk / jnp.sum(nk))
        nkl = nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    mp = jnp.einsum("iz,jz,kz,lz,mz,ijoklm->ioklm", *facBuild.factors, tPrecision)

    n_markers = tPrecision.shape[1]
    Xp = jnp.einsum("ji...,njo...->ino...", X, tPrecision)
    log_prob = jnp.square(jnp.linalg.norm(Xp - mp[jnp.newaxis, ...], axis=2))
    log_prob = -0.5 * (n_markers * jnp.log(2 * jnp.pi) + log_prob) + nkl

    log_sum = jsp.logsumexp(log_prob, axis=1)
    log_resp = log_prob - log_sum[:, np.newaxis, ...]
    return jnp.exp(log_resp)


def maxll(
    vec: jnp.ndarray,
    shape: tuple[int, ...],
    X: np.ndarray,
    nk_r: bool = False,
) -> jnp.ndarray:
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    facBuild = tensorGMM(vec, shape, nk_rearrange=nk_r)

    # Creating function that we want to minimize
    return -comparingGMMjax(X, facBuild) / X.shape[1]


def minimize_func(
    X: xa.DataArray,
    rank: int,
    n_cluster: int,
    maxiter: int = 8000,
    verbose: bool = True,
    x0=None,
    nk_rearrange: bool = False,
    seed=None,
):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    meanShape = (n_cluster, X.shape[0], *X.shape[2::])

    args = (meanShape, X.to_numpy(), nk_rearrange)
    func = jit(value_and_grad(maxll), static_argnums=(1, 3))

    def ffunc(*arrgs):
        a, b = func(*arrgs)
        return a, np.array(b, dtype=float)

    if x0 is None:
        x0 = vector_guess(meanShape, rank, seed=seed, nk_rearrange=nk_rearrange)

    tq = tqdm(total=maxiter, delay=0.1, disable=(verbose is False))

    def callback(xk):
        a, b = func(xk, *args)
        gNorm = np.linalg.norm(b)
        tq.set_postfix(val="{:.2e}".format(a), g="{:.2e}".format(gNorm), refresh=False)
        tq.update(1)

    opts = {"maxiter": maxiter, "disp": False}

    # Add bounds
    lb = np.full_like(x0, -np.inf)
    ub = copy(-lb)
    if nk_rearrange:
        lb[0:n_cluster * rank] = 0.1
        ub[0:n_cluster * rank] = 10
    else:
        lb[0:n_cluster] = 0.1
        ub[0:n_cluster] = 10

    bounds = Bounds(lb, ub, keep_feasible=False)


    opt = minimize(
        ffunc,
        x0,
        jac=True,
        callback=callback,
        method="L-BFGS-B",
        bounds=bounds,
        args=args,
        options=opts,
    )
    tq.close()

    return tensorGMM(opt.x, meanShape, nk_rearrange), -opt.fun, opt.x


def tensorGMM_CV(X, numFolds: int, numClusters: int, numRank: int, maxiter: int = 200):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0.0
    meanShape = (numClusters, X.shape[0], X.shape[2], X.shape[3], X.shape[4])

    kf = KFold(n_splits=numFolds)
    x0 = None

    # Start generating splits and running model
    for train_index, test_index in kf.split(X[:, :, 0, 0, 0].T):
        # Train
        _, _, x0 = minimize_func(
            X[:, train_index, :, :, :],
            numRank,
            numClusters,
            maxiter=maxiter,
            verbose=False,
            x0=x0,
        )
        # Test
        test_ll = -maxll(x0, meanShape, X[:, test_index, :, :, :].to_numpy())
        logLik += test_ll

    return float(logLik)


def optimal_seed(n_seeds, *args, **kwargs):
    """Finds the optimal seed number to minimize log likeihood"""
    fits = [minimize_func(*args, **kwargs, seed=i) for i in range(n_seeds)]
    total_loglik = [f[1] for f in fits]
    best_seed = np.argmax(total_loglik)
    return best_seed, total_loglik[best_seed], fits[best_seed]
