"""
Test the data import.
"""
import numpy as np
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from ..parafac2 import parafac2_nd, _cmf_reconstruction_error
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error


pf2shape = [(100, 800)] * 4
X = random_parafac2(pf2shape, rank=3, full=True, random_state=2)
norm_tensor = np.linalg.norm(X) ** 2


def test_parafac2():
    """Test for equivalence to TensorLy's PARAFAC2."""
    w1, f1, p1, e1 = parafac2_nd(X, rank=3, random_state=1, verbose=False)

    # Test that the model still matches the data
    err = _parafac2_reconstruction_error(X, (w1, f1, p1)) ** 2
    np.testing.assert_allclose(1.0 - err / norm_tensor, e1, rtol=1e-6)

    # Test reproducibility
    w2, f2, p2, e2 = parafac2_nd(X, rank=3, random_state=1, verbose=False)
    # Compare to TensorLy
    wT, fT, pT = parafac2(
        X,
        rank=3,
        normalize_factors=True,
        n_iter_max=5,
        init=(w1.copy(), [f.copy() for f in f1], [p.copy() for p in p1]),
    )

    # Check normalization
    for ff in [f1, f2, fT]:
        for ii in range(3):
            np.testing.assert_allclose(np.linalg.norm(ff[ii], axis=0), 1.0, rtol=1e-2)

    # Compare both seeds
    np.testing.assert_allclose(w1, w2)
    np.testing.assert_allclose(e1, e2)
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], f2[ii])
        np.testing.assert_allclose(p1[ii], p2[ii])

    # Compare to TensorLy
    np.testing.assert_allclose(w1, wT, rtol=0.01)
    for ii in range(3):
        np.testing.assert_allclose(f1[ii], fT[ii], rtol=0.01, atol=0.01)
        np.testing.assert_allclose(p1[ii], pT[ii], rtol=0.01, atol=0.01)


def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    w, f, _ = random_parafac2(pf2shape, rank=3, random_state=1, normalise_factors=False)

    errCMF, p, _ = _cmf_reconstruction_error(X, f, norm_tensor)
    err = _parafac2_reconstruction_error(X, (w, f, p)) ** 2

    np.testing.assert_allclose(err, errCMF, rtol=1e-12)
