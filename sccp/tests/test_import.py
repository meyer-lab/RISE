"""
Test the cross validation accuracy.
"""
import numpy as np
import pytest
from ..imports import import_thomson, import_lupus, import_citeseq
from ..gating import gateThomsonCells


def test_Thomson():
    """Test for correctness of cross validation."""
    X = import_thomson()
    print()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32


@pytest.mark.skip("The lupus dataset uses too much memory for now.")
def test_Lupus():
    """Test for correctness of cross validation."""
    X = import_lupus()


def test_CITEseq():
    """Test for correctness of cross validation."""
    X = import_citeseq()
    print()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32


def test_GateThomson():
    """Test that gating function matches size of data"""
    X = import_thomson()
    gateThomsonCells(X)