"""
Test the data import.
"""
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from tensorly.random import random_cp
from ..imports import smallDF
from ..GMM import cvGMM, probGMM
from ..tensor import cp_to_vector, vector_to_cp, comparingGMM, comparingGMMjax

data_import, other_import = smallDF(10)


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    cp_tensor = random_cp((10, 11, 12, 13, 14), 3, normalise_factors=False)
    cpVector = cp_to_vector(cp_tensor)
    vectorFac = vector_to_cp(cpVector, cp_tensor.rank, cp_tensor.shape)

    for ii in range(len(vectorFac.factors)):
        assert_allclose(vectorFac.factors[ii], cp_tensor.factors[ii])


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    nk, tMeans, tCovar = probGMM(data_import, 2)
    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))

    optimized1 = comparingGMM(data_import, tMeans, tCovar.to_numpy(), nkValues)
    optimized2 = comparingGMMjax(data_import, tMeans, tCovar.to_numpy(), nkValues)

    np.testing.assert_almost_equal(optimized1, optimized2)
