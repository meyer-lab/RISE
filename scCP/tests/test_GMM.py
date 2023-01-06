"""
Test the data import.
"""
import pytest
import numpy as np
from ..imports.cytok import smallDF
from ..imports.CoH import CoH_xarray


data_import, other_import = smallDF(10)
meanShape = (
    6,
    data_import.shape[0],
    data_import.shape[2],
    data_import.shape[3],
    data_import.shape[4],
)


def test_import():
    """Stub test."""
    dataTwo, _ = smallDF(data_import.shape[1] * 2)
    assert data_import.shape[0] == dataTwo.shape[0]
    assert 2 * data_import.shape[1] == dataTwo.shape[1]
    assert data_import.shape[2] == dataTwo.shape[2]
    assert data_import.shape[3] == dataTwo.shape[3]
    assert data_import.shape[4] == dataTwo.shape[4]


@pytest.mark.parametrize("cells", [5, 100])
def test_import_CoH(cells):
    """Test the CoH import."""
    cond = ["Untreated", "IFNg-50ng", "IL10-50ng", "IL4-50ng", "IL2-50ng", "IL6-50ng"]
    numCell = cells
    cohXA_import, _, _ = CoH_xarray(numCell, cond, allmarkers=True)
    assert np.isfinite(cohXA_import.to_numpy()).all()


def test_finite_data():
    """Test that all values in tensor has no NaN"""
    assert np.isfinite(data_import.to_numpy()).all()

