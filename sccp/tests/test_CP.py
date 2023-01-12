"""
Test the data import.
"""
import tensorly as tl
import numpy as np
from ..parafac2 import (
    parafac2,
    _project_tensor_slices_fused,
    _compute_projections_fused,
)
from tensorly.decomposition import parafac2 as pf2
from tensorly.random import random_cp
from tensorly.decomposition._parafac2 import (
    _project_tensor_slices,
    _compute_projections,
)


def test_n_way():
    """Stub test."""
    rcp = random_cp((10, 50, 30), rank=3, full=True)
    rcpStack = np.stack([rcp, rcp, rcp, rcp, rcp, rcp], axis=0)

    _, factors, _ = pf2(rcp, rank=3, normalize_factors=True)
    _, facStack, _ = parafac2(rcpStack, rank=3, n_iter_max=10, tol=1e-10)

    # assert np.testing.assert_allclose(factors[0], facStack[0])


def test_proj_tensor():
    """Stub test."""
    X = random_cp((10, 50, 30), rank=3, full=True)
    projections = np.random.randn(10, 50, 3)

    p_t = _project_tensor_slices(X, projections)
    p_t_fused = _project_tensor_slices_fused(X, projections)

    np.testing.assert_allclose(p_t, p_t_fused)


def test_comp_proj():
    """Stub test."""
    X = random_cp((10, 50, 30), rank=3, full=True)
    factors = [np.random.randn(10, 3), np.random.randn(3, 3), np.random.randn(30, 3)]

    projs = _compute_projections(X, factors, tl.partial_svd)
    proj_fused = _compute_projections_fused(X, factors)
    projs = np.stack(projs, axis=0)

    np.testing.assert_allclose(projs, proj_fused)
