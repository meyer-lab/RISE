"""
Test the parafac2 method.
"""

import numpy as np
import pandas as pd

from analysis.imports import import_thomson

from ..factorization import pf2, rise_pca_r2x


def test_factor_thomson_reprod():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    res1 = pf2(X.copy(), 5, doEmbedding=False, tolerance=1e-6)
    C_first = np.array(res1.varm["Pf2_C"], copy=True)

    res2 = pf2(X.copy(), 5, doEmbedding=False, tolerance=1e-6)
    np.testing.assert_allclose(
        np.array(res2.varm["Pf2_C"]), C_first, atol=1e-5, rtol=1e-5
    )


def test_factor_thomson_R2X():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    r2x_rise, r2x_pca = rise_pca_r2x(X, np.arange(1, 4))
    print(r2x_pca)
    print(r2x_rise)

    assert np.all(r2x_rise > np.array([0.0001, 0.0002, 0.0003]))
