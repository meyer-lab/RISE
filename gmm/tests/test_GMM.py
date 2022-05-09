"""
Test the data import.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from ..imports import smallDF
from ..GMM import cvGMM, probGMM
from ..tensor import cp_pt_to_vector, vector_to_cp_pt, comparingGMM, comparingGMMjax

data_import, other_import = smallDF(10)


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    rand_vec = np.random.random(2130)

    built = vector_to_cp_pt(rand_vec, 3, (6, 5, 4, 12, 8))
    out_vec = cp_pt_to_vector(built[0], built[2])

    assert_allclose(rand_vec, out_vec)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    nk, tMeans, tCovar = probGMM(data_import, 2)
    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))

    optimized1 = comparingGMM(data_import, tMeans, tCovar.to_numpy(), nkValues)
    optimized2 = comparingGMMjax(data_import.to_numpy(), tMeans.to_numpy(), tCovar.to_numpy(), nkValues)

    np.testing.assert_almost_equal(optimized1, optimized2)
