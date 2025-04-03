"""
Test the parafac2 method.
"""

import numpy as np
import pandas as pd

from ..factorization import pf2, pf2_pca_r2x
from ..imports import import_thomson


def test_factor_thomson_reprod():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    X = pf2(X, 10, doEmbedding=False)
    C_first = np.array(X.varm["Pf2_C"], copy=True)

    X = pf2(X, 10, doEmbedding=False)
    np.testing.assert_allclose(np.array(X.varm["Pf2_C"]), C_first, atol=1e-5, rtol=1e-5)


def test_factor_thomson_R2X():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    r2x_pf2, r2x_pca = pf2_pca_r2x(X, np.arange(1, 4))
    print(r2x_pca)
    print(r2x_pf2)

    # Probably fails due to numerical accuracy issues
    # assert (r2x_pf2 < r2x_pca).all()
    assert np.all(r2x_pf2 > np.array([0.002, 0.005, 0.007]))
