"""
Test the data import.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from ..imports import smallDF
from ..GMM import cvGMM, probGMM
from ..tensor import cp_pt_to_vector, vector_to_cp_pt, comparingGMM, comparingGMMjax, vector_guess, maxloglik_ptnnp

data_import, other_import = smallDF(10)


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    meanShape = (6, 5, 4, 12, 8)
    x0 = vector_guess(meanShape, rank=3)

    built = vector_to_cp_pt(x0, 3, meanShape)
    out_vec = cp_pt_to_vector(*built)

    # Check that we can get a likelihood
    ll = maxloglik_ptnnp(x0, meanShape, 3, data_import)

    assert np.isfinite(ll)
    assert_allclose(x0, out_vec)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    nk, tMeans, tCovar = probGMM(data_import, 2)
    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))

    optimized1 = comparingGMM(data_import, tMeans, tCovar.to_numpy(), nkValues)
    optimized2 = comparingGMMjax(data_import.to_numpy(), tMeans.to_numpy(), tCovar.to_numpy(), nkValues)

    np.testing.assert_almost_equal(optimized1, optimized2)
