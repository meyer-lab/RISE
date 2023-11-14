"""
Test the parafac2 method.
"""
import numpy as np
from ..parafac2 import pf2, pf2_r2x
from ..imports import import_thomson


def test_factor_thomson():
    """Import and factor Thomson."""
    X = import_thomson()

    X = pf2(X, 30, doEmbedding=False)

    r2x = pf2_r2x(X, 4)
    assert np.all(r2x > np.array([[0.013, 0.016, 0.018, 0.02]]))
    print(r2x)
