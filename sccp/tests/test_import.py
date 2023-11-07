"""
Test the cross validation accuracy.
"""
import pytest
import numpy as np
from ..imports import import_thomson, import_lupus, import_citeseq
from ..gating import gateThomsonCells


@pytest.mark.parametrize("import_func", [import_thomson, import_lupus, import_citeseq])
def test_imports(import_func):
    """Test import functions."""
    X = import_func()
    print()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32


def test_GateThomson():
    """Test that gating function matches size of data"""
    X = import_thomson()
    gateThomsonCells(X)
