"""
Test the parafac2 method.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.imports import import_thomson

from ..factorization import pf2, rise_pca_r2x


def test_factor_thomson_R2X():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    r2x_rise, r2x_pca = rise_pca_r2x(X, np.arange(1, 4))
    print(r2x_pca)
    print(r2x_rise)

    assert np.all(r2x_rise > np.array([0.0001, 0.0002, 0.0003]))


def test_factor_thomson_mlx_backend():
    """Run the factorization on the MLX (Apple GPU) backend.

    Skipped unless MLX is installed, which is only the case on the
    macOS/arm64 CI runner where the `gpu` extra is installed.
    """
    pytest.importorskip("mlx.core")

    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    res = pf2(X, 5, doEmbedding=False, tolerance=1e-6, backend="mlx")

    assert np.all(np.isfinite(res.varm["Pf2_C"]))
