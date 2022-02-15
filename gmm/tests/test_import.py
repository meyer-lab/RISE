'''
Test the data import.
'''
import pytest
from ..imports import smallDF


def test_import():
    """ Stub test. """
    data = smallDF(10)
    dataTwo = smallDF(20)
    assert 2 * data.shape[0] == dataTwo.shape[0]
