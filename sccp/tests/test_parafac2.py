"""
Test the parafac2 method.
"""
import anndata
from scipy.sparse import random, csr_array
import numpy as np
import cupy as cp
import tensorly as tl
from tensorly.random import random_parafac2
from tensorly.parafac2_tensor import (
    parafac2_to_slices,
)
from ..parafac2 import pf2, pf2_r2x, calc_total_norm, _cmf_reconstruction_error
from ..imports import import_thomson


def test_total_norm():
    """This tests that mean centering does not affect the projections and error calculation."""
    X = anndata.AnnData(X=random(200, 200, density=0.1, format="csr"))  # type: ignore
    X.var["means"] = np.zeros(X.shape[1])

    normBefore = calc_total_norm(X)

    # De-mean since we aim to subtract off the means
    means = np.mean(X.X.toarray(), axis=0)  # type: ignore
    X.X += means
    X.var["means"] = means

    normAfter = calc_total_norm(X)
    np.testing.assert_allclose(normBefore, normAfter)


def test_pf2_proj_centering():
    """Test that centering the matrix does not affect the results."""
    tl.set_backend("cupy")
    _, factors, projections = random_parafac2(
        shapes=[(25, 300) for _ in range(15)],
        rank=3,
        normalise_factors=False,
        dtype=tl.float64,
    )

    X_pf = parafac2_to_slices((None, factors, projections))

    XX = cp.asnumpy(cp.concatenate(X_pf, axis=0))
    means = cp.zeros(XX.shape[1])
    sgIndex = np.repeat(np.arange(15), 25)
    assert sgIndex.shape[0] == XX.shape[0]

    norm_X_sq = float(np.linalg.norm(XX) ** 2.0)

    norm_sq_err, _, _ = _cmf_reconstruction_error(
        csr_array(XX), sgIndex, means, factors, norm_X_sq
    )
    np.testing.assert_allclose(norm_sq_err.get() / norm_X_sq, 0.0, atol=1e-6)

    # De-mean since we aim to subtract off the means
    means = np.mean(XX, axis=0)
    XX = XX + means

    norm_sq_err_centered, _, _ = _cmf_reconstruction_error(
        csr_array(XX), sgIndex, cp.array(means), factors, norm_X_sq
    )
    np.testing.assert_allclose(
        norm_sq_err.get() / norm_X_sq, norm_sq_err_centered.get() / norm_X_sq, atol=1e-6
    )

    tl.set_backend("numpy")


def test_factor_thomson():
    """Import and factor Thomson."""
    X = import_thomson()

    X = pf2(X, 30, doEmbedding=False)

    r2x = pf2_r2x(X, 4)
    assert np.all(r2x > np.array([[0.012, 0.015, 0.017, 0.018]]))
    print(r2x)
