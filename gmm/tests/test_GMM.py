"""
Test the data import.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from ..imports import smallDF
from ..GMM import cvGMM
from tensorly.tenalg import multi_mode_dot
from ..tensor import cp_pt_to_vector, vector_to_cp_pt, comparingGMM, comparingGMMjax, vector_guess, maxloglik_ptnnp, minimize_func

data_import, other_import = smallDF(10)


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    meanShape = (6, 5, 4, 12, 8)
    x0 = vector_guess(meanShape, rank=3)

    built = vector_to_cp_pt(x0, 3, meanShape, enforceSPD=False)
    vector_to_cp_pt(x0, 3, meanShape, enforceSPD=True)
    out_vec = cp_pt_to_vector(*built)

    # Check that we can get a likelihood
    ll = maxloglik_ptnnp(x0, meanShape, 3, data_import.to_numpy())

    assert np.isfinite(ll)
    assert_allclose(x0, out_vec)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    meanShape = (6, 5, 4, 12, 8)
    x0 = vector_guess(meanShape, rank=3)

    nk, meanFact, ptFact, ptCore = vector_to_cp_pt(x0, 3, meanShape)
    ptCoreFull = np.einsum("ijk,lkmno->lijmno", ptFact[1], ptCore)
    ptBuilt = multi_mode_dot(ptCoreFull, [ptFact[0], ptFact[2], ptFact[3], ptFact[4]], modes=[0, 3, 4, 5], transpose=False)
    ptBuilt = (ptBuilt + np.swapaxes(ptBuilt, 1, 2)) / 2.0  # Enforce symmetry

    optimized1 = comparingGMM(data_import, meanFact, ptBuilt, nk)
    optimized2 = comparingGMMjax(data_import.to_numpy(), nk, meanFact, ptFact, ptCore)

    np.testing.assert_almost_equal(optimized1, optimized2)


def test_fit():
    """Test that fitting can run fine."""
    nk, fac, core, ll = minimize_func(data_import, 2, 3, maxiter=500)
