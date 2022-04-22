"""
Test the data import.
"""
import pandas as pd
from numpy.testing import assert_allclose
from tensorly.random import random_cp
from ..imports import smallDF
from ..GMM import cvGMM, probGMM
from ..tensor import cp_to_vector, vector_to_cp


def test_cvGMM():
    """Stub test."""
    dataTwo, other = smallDF(50)
    gmmDF = cvGMM(dataTwo, 4, other[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_GMMprob():
    """Test that we can construct a covariance matrix including pSTAT5."""
    cellperexp = 50
    dataTwo, _ = smallDF(cellperexp)
    maxcluster = 4
    nk, means, covari = probGMM(dataTwo, maxcluster)


def test_TensorVector():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    cp_tensor = random_cp((10, 11, 12, 13, 14), 3, normalise_factors=False)
    cpVector = cp_to_vector(cp_tensor)
    vectorFac = vector_to_cp(cpVector, cp_tensor.rank, cp_tensor.shape)

    for ii in range(len(vectorFac.factors)):
        assert_allclose(vectorFac.factors[ii], cp_tensor.factors[ii])
