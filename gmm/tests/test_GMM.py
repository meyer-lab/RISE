"""
Test the data import.
"""
import pandas as pd
import numpy as np
from ..imports import smallDF
from ..GMM import cvGMM
from ..scImport import import_thompson_drug
from ..tensor import vector_to_cp_pt, comparingGMM, comparingGMMjax, vector_guess, maxloglik_ptnnp, minimize_func, tensorGMM_CV, covFactor_to_precisions

data_import, other_import = smallDF(10)
meanShape = (6, data_import.shape[0], data_import.shape[2], data_import.shape[3], data_import.shape[4])


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


def test_sc():
    x, _ = import_thompson_drug()


def test_cov_to_prec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    _, _, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    assert np.all(np.isfinite(precBuild))


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    built = vector_to_cp_pt(x0, 3, meanShape)

    # Check that we can get a likelihood
    ll = maxloglik_ptnnp(x0, meanShape, 3, data_import.to_numpy())

    assert np.isfinite(ll)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    x0 = vector_guess(meanShape, rank=3)

    nk, meanFact, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    optimized1 = comparingGMM(data_import, meanFact, precBuild, nk)
    optimized2 = comparingGMMjax(data_import.to_numpy(), nk, meanFact, precBuild)
    np.testing.assert_allclose(optimized1, optimized2, rtol=1e-5)


def test_independence():
    """Test that conditions can be separately evaluated as expected."""
    x0 = vector_guess(meanShape, rank=3)
    data_numpy = data_import.to_numpy()

    nk, meanFact, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    ll1 = comparingGMM(data_import, meanFact, precBuild, nk)
    ll2 = comparingGMMjax(data_numpy, nk, meanFact, precBuild)
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)

    # Test that cells are independent
    ll3 = comparingGMMjax(data_numpy[:, :5, :, :, :], nk, meanFact, precBuild)
    ll3 += comparingGMMjax(data_numpy[:, 5:, :, :, :], nk, meanFact, precBuild)
    np.testing.assert_allclose(ll2, ll3, rtol=1e-5)

    # Test that ligands are independent
    # meanFactOne = deepcopy(meanFact)
    # meanFactOne[4] = meanFact[4][:5, :]
    # ptFactOne = deepcopy(ptFact)
    # ptFactOne[4] = ptFact[4][:5, :]
    # ll4 = comparingGMMjax(data_numpy[:, :, :, :, :5], nk, meanFactOne, ptFactOne)
    # meanFactTwo = deepcopy(meanFact)
    # meanFactTwo[4] = meanFact[4][5:, :]
    # ptFactTwo = deepcopy(ptFact)
    # ptFactTwo[4] = ptFact[4][5:, :]
    # ll4 += comparingGMMjax(data_numpy[:, :, :, :, 5:], nk, meanFactTwo, ptFactTwo)
    # np.testing.assert_allclose(ll2, ll4, rtol=1e-5)


def test_fit():
    """Test that fitting can run fine."""
    nk, fac, ptfac, ll, _, _ = minimize_func(data_import, 3, 10, maxiter=20)
    loglik = tensorGMM_CV(data_import, numFolds=3, numClusters=3, numRank=2, maxiter=20)
    assert isinstance(loglik, float)
    assert isinstance(ll, float)
