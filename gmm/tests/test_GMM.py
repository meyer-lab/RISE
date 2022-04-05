"""
Test the data import.
"""
import pytest
import pandas as pd
from ..imports import smallDF
from ..GMM import cvGMM, probGMM


def test_import():
    """Stub test."""
    (
        dataTwo,
        _,
    ) = smallDF(50)
    gmmDF = cvGMM(dataTwo, 4)
    assert isinstance(gmmDF, pd.DataFrame)


def test_GMMprob():
    """Test that we can construct a covariance matrix including pSTAT5."""
    cellperexp = 50
    dataTwo, _ = smallDF(cellperexp)
    nk, means, covari = probGMM(dataTwo, 4)
