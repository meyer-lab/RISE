"""
Test the data import.
"""
from ..imports import smallDF


def test_import():
    """Stub test."""
    data, _ = smallDF(10)
    dataTwo, _ = smallDF(20)
    assert data.shape[0] == dataTwo.shape[0]
    assert 2 * data.shape[1] == dataTwo.shape[1]
    assert data.shape[2] == dataTwo.shape[2]
    assert data.shape[3] == dataTwo.shape[3]
    assert data.shape[4] == dataTwo.shape[4]
