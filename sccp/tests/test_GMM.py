"""
Test the data import.
"""
import pytest
import numpy as np
from ..imports.cytok import IL2_flowXA
from ..imports.CoH import CoH_xarray

IL2data_import, _= IL2_flowXA()

@pytest.mark.parametrize("cells", [5, 100])
def test_import_CoH(cells):
    """Test the CoH import."""
    cond = ["Untreated", "IFNg-50ng", "IL10-50ng", "IL4-50ng", "IL2-50ng", "IL6-50ng"]
    numCell = cells
    cohXA_import, _, _ = CoH_xarray(numCell, cond, allmarkers=True)
    assert np.isfinite(cohXA_import.to_numpy()).all()

def test_finite_data():
    """Test that all values in tensor has no NaN"""
    assert np.isfinite(IL2data_import.to_numpy()).all()

