"""
Test the cross validation accuracy.
"""
import numpy as np
from ..imports.scRNA import import_thomson, import_lupus, import_citeseq


def test_Thomson():
    """Test for correctness of cross validation."""
    X = import_thomson()

    assert np.all(np.isfinite(X.X))


def test_Lupus():
    """Test for correctness of cross validation."""
    X = import_lupus()

    assert np.all(np.isfinite(X.X))


def test_CITEseq():
    """Test for correctness of cross validation."""
    X = import_citeseq()
