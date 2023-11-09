"""
Test the parafac2 method.
"""
from ..parafac2 import pf2, pf2_r2x
from ..imports import import_thomson


def test_factor_thomson():
    """Import and factor Thomson."""
    X = import_thomson()

    X = pf2(X, 3, doEmbedding=False)

    r2x = pf2_r2x(X, 4)
    print(r2x)
