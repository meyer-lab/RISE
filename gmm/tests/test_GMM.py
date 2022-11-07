"""
Test the data import.
"""
import pytest
import pandas as pd
import numpy as np
import xarray as xa
import tensorly as tl
from sklearn.mixture import GaussianMixture
from ..imports import smallDF
from ..GMM import cvGMM
from ..scImport import ThompsonDrugXA
from ..CoHimport import CoH_xarray
from ..tensor import (
    comparingGMMjax,
    vector_guess,
    maxll,
    minimize_func,
    tensorGMM_CV,
    tensorGMM,
    infer_rank,
    norm_covariances,
)

data_import, other_import = smallDF(10)
meanShape = (
    6,
    data_import.shape[0],
    data_import.shape[2],
    data_import.shape[3],
    data_import.shape[4],
)


def comparingGMM(zflowDF: xa.DataArray, facs: tensorGMM):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    nk = facs.nk
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    tMeans = tl.cp_to_tensor(facs)
    tPrecision = facs.get_precisions()
    X = zflowDF.to_numpy()

    it = np.nditer(tMeans[0, 0, :, :, :], flags=["multi_index", "refs_ok"])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        Xcur = np.transpose(X[:, :, i, j, k])  # Cell Number per experiment x Marker

        if np.all(np.isnan(Xcur)):  # Skip if there's no data
            continue

        gmm = GaussianMixture(
            n_components=nk.size,
            covariance_type="full",
            means_init=tMeans[:, :, i, j, k],
            weights_init=nk,
        )
        gmm._initialize(Xcur, np.ones((X.shape[1], nk.size)))  # Markers x Clusters
        gmm.precisions_cholesky_ = tPrecision[
            :, :, :, i, j, k
        ]  # Cluster x Marker x Marker
        loglik += np.sum(gmm.score_samples(Xcur))

    return loglik


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_import():
    """Stub test."""
    dataTwo, _ = smallDF(data_import.shape[1] * 2)
    assert data_import.shape[0] == dataTwo.shape[0]
    assert 2 * data_import.shape[1] == dataTwo.shape[1]
    assert data_import.shape[2] == dataTwo.shape[2]
    assert data_import.shape[3] == dataTwo.shape[3]
    assert data_import.shape[4] == dataTwo.shape[4]


def test_cov_to_prec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    facs = tensorGMM(x0, meanShape)
    precBuild = facs.get_precisions()

    assert np.all(np.isfinite(precBuild))


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    facs = tensorGMM(x0, meanShape)

    # Check that we can get a likelihood
    ll = maxll(x0, meanShape, data_import.to_numpy())

    assert np.isfinite(ll)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    x0 = vector_guess(meanShape, rank=3)

    facs = tensorGMM(x0, meanShape)

    optimized1 = comparingGMM(data_import, facs)
    optimized2 = comparingGMMjax(data_import.to_numpy(), facs)
    np.testing.assert_allclose(optimized1, optimized2, rtol=1e-5)


def test_independence():
    """Test that conditions can be separately evaluated as expected."""
    x0 = vector_guess(meanShape, rank=3)
    data_numpy = data_import.to_numpy()

    facs = tensorGMM(x0, meanShape)

    ll1 = comparingGMM(data_import, facs)
    ll2 = comparingGMMjax(data_numpy, facs)
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)

    # Test that cells are independent
    ll3 = comparingGMMjax(data_numpy[:, :5, :, :, :], facs)
    ll3 += comparingGMMjax(data_numpy[:, 5:, :, :, :], facs)
    np.testing.assert_allclose(ll2, ll3, rtol=1e-5)

    # Test that ligands are independent
    # meanFactOne = deepcopy(meanFact)
    # meanFactOne[4] = meanFact[4][:5, :]
    # ptFactOne = deepcopy(ptFact)
    # ptFactOne[4] = ptFact[4][:5, :]
    # ll4 = comparingGMMjax(data_numpy[:, :, :, :, :5], facs)
    # meanFactTwo = deepcopy(meanFact)
    # meanFactTwo[4] = meanFact[4][5:, :]
    # ptFactTwo = deepcopy(ptFact)
    # ptFactTwo[4] = ptFact[4][5:, :]
    # ll4 += comparingGMMjax(data_numpy[:, :, :, :, 5:], facs)
    # np.testing.assert_allclose(ll2, ll4, rtol=1e-5)


@pytest.mark.parametrize("nk_r", [True, False])
def test_fit(nk_r):
    """Test that fitting can run fine."""
    facs, ll, _ = minimize_func(
        data_import, 3, 10, maxiter=20, seed=1, verbose=False, nk_rearrange=nk_r
    )
    facsTwo, llTwo, _ = minimize_func(
        data_import, 3, 10, maxiter=20, seed=1, verbose=False, nk_rearrange=nk_r
    )
    loglik = tensorGMM_CV(data_import, numFolds=3, numClusters=3, numRank=2, maxiter=20)
    assert isinstance(loglik, float)
    assert isinstance(ll, float)
    np.testing.assert_allclose(ll, llTwo)
    np.testing.assert_allclose(facs.nk, facsTwo.nk)


@pytest.mark.parametrize("rank", [3, 10])
def test_import_PopAlign(rank):
    """Test the scRNAseq import."""
    dataPA_import, _, _, _ = ThompsonDrugXA(rank=rank)
    assert dataPA_import.shape == (rank, 290, 46, 1, 1)
    assert np.isfinite(dataPA_import.to_numpy()).all()


@pytest.mark.parametrize("cells", [5, 100])
def test_import_CoH(cells):
    """Test the CoH import."""
    cond = ["Untreated", "IFNg-50ng", "IL10-50ng", "IL4-50ng", "IL2-50ng", "IL6-50ng"]
    numCell = cells
    cohXA_import, _, _ = CoH_xarray(numCell, cond, allmarkers=True)
    assert np.isfinite(cohXA_import.to_numpy()).all()


def test_finite_data():
    """Test that all values in tensor has no NaN"""
    assert np.isfinite(data_import.to_numpy()).all()


@pytest.mark.parametrize("rank", [3, 10])
@pytest.mark.parametrize("nk_r", [True, False])
def test_infer_rank(rank, nk_r):
    """Test that we correctly infer the vector rank."""
    x0 = vector_guess(meanShape, rank=rank, nk_rearrange=nk_r)
    rank_inf = infer_rank(x0.size, meanShape, nk_rearrange=nk_r)
    assert rank_inf == rank

def test_cov_tensor():
    """Test that covariance factor normalization does not affect the reconstruction."""
    fac, _, _ = minimize_func(data_import, rank=3, n_cluster=5, maxiter=10, seed=1, verbose=False)

    # Perturb covFacs
    fac.covFacs[0] *= 2.0
    fac.covFacs[1] *= 5.0
    fac.covars /= 3.0

    covTensor1 = np.array(fac.get_covariances())

    fac.covars, fac.covFacs, fac.covWeights = norm_covariances(fac.covars, fac.covFacs, fac.covWeights)

    covTensor2 = np.array(fac.get_covariances())
    np.testing.assert_allclose(covTensor1, covTensor2, rtol=1e-6)


def test_cov_normalization():
    """Test that running the covariance normalization a second time creates the same result."""
    # Note that the first covariance normalization happens as part of the fitting process.
    fac, _, _ = minimize_func(data_import, rank=4, n_cluster=6, maxiter=10, seed=1, verbose=False)

    covars, covFacs, _ = norm_covariances(fac.covars, fac.covFacs)

    np.testing.assert_allclose(fac.covFacs[0], covFacs[0], rtol=1e-6)
    np.testing.assert_allclose(fac.covars, covars, rtol=1e-6)
