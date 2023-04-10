"""
Test the data import.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from tensorly.metrics import correlation_index
from ..parafac2 import parafac2_nd, _cmf_reconstruction_error
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error


pf2shape = [(10, 30)] * 4
X = random_parafac2(pf2shape, rank=3, full=True, random_state=2)


def test_parafac2():
    """Test for equivalence to TensorLy's PARAFAC2."""
    _, factors, pTensorly = parafac2(X, rank=3, normalize_factors=True, init="svd")
    w1, f1, p1, _ = parafac2_nd(X, rank=3, random_state=1)

    # More similar is closer to 0 with corrIndex
    assert correlation_index(factors, f1, method="min_score") < 0.1

    # Compare projection matrices, too
    assert correlation_index(pTensorly, p1, method="min_score") < 0.1

    # Test reproducibility
    w2, f2, _, _ = parafac2_nd(X, rank=3, random_state=1)
    np.testing.assert_almost_equal(w1, w2)
    np.testing.assert_almost_equal(f1[0], f2[0])


def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    w, f, _ = random_parafac2(pf2shape, rank=3, random_state=1, normalise_factors=False)

    norm_tensor = tl.norm(X) ** 2

    errCMF, p = _cmf_reconstruction_error(X, f, norm_tensor)
    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_almost_equal(err, errCMF)
