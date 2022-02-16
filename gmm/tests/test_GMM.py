'''
Test the data import.
'''
import pytest
import pandas as pd
from ..imports import smallDF
from ..GMM import GMMpca


@pytest.mark.parametrize("score", [None, "rand_score"])
def test_import(score):
    """ Stub test. """
    dataTwo = smallDF(5)
    gmmDF = GMMpca(dataTwo, 5, score)
    assert isinstance(gmmDF, pd.DataFrame)
