'''
Test the data import.
'''
import pytest
import pandas as pd
from ..imports import smallDF
from ..GMM import GMMpca


def test_import():
    """ Stub test. """
    dataTwo = smallDF(50)
    gmmDF = GMMpca(dataTwo, 5)
    assert isinstance(gmmDF, pd.DataFrame)
