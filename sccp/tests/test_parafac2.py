"""
Test the parafac2 method.
"""

import numpy as np
from ..factorization import pf2, pf2_pca_r2x
from ..imports import import_thomson


def test_factor_thomson():
    """Import and factor Thomson.
    This also checks that the factorization process is reproducible."""
    X = import_thomson()

    X = pf2(X, 10, doEmbedding=False)
    C_first = np.array(X.varm["Pf2_C"], copy=True)

    X = pf2(X, 10, doEmbedding=False)
    np.testing.assert_allclose(np.array(X.varm["Pf2_C"]), C_first, atol=1e-5, rtol=1e-5)

    r2x = pf2_pca_r2x(X, np.array([1, 2, 3]))
    assert np.all(r2x[0] > np.array([0.002, 0.005, 0.007]))
    print(r2x)
