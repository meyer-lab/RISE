"""
Test the data import.
"""
import pytest
import numpy as np
from ..imports.cytok import IL2_flowXA

IL2data_import, _ = IL2_flowXA()

def test_finite_data():
    """Test that all values in tensor has no NaN"""
    assert np.isfinite(IL2data_import.to_numpy()).all()
