"""
Test the data import.
"""
import numpy as np
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from tensorly.metrics import correlation_index
from ..parafac2 import parafac2_nd
from ..crossVal import crossvalidate
from ..imports.scRNA import ThompsonXA_SCGenes
from ..crossVal import crossvalidate
from tensorly.decomposition._parafac2 import _parafac2_reconstruction_error
import tensorly as tl

def test_parafac():
    """Test for equivalence to TensorLy's PARAFAC2."""
    X = random_parafac2([(10, 30)] * 5, rank=3, full=True, random_state=1)
    X = np.stack(X, axis=0)

    _, factors, projections = parafac2(X, rank=3, normalize_factors=True, init="svd")
    _, facStack, projStack, _ = parafac2_nd(X, rank=3)

    # More similar is closer to 0 with corrIndex
    assert correlation_index(factors, facStack, method="min_score") < 0.1

    # Compare projection matrices, too
    assert (
        correlation_index(list(projections), list(projStack), method="min_score") < 0.1
    )


def test_pf2_speed():
    """Compare run time for different SVD initialization"""
    drugXA = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    X = drugXA["data"].to_numpy()

    crossvalidate(X, rank=3)

    _, _, _, _ = parafac2_nd(X, rank=4, verbose=True)
    

def test_pf2_speed():
    """Compare run time for different SVD initialization"""
    drugXA = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    X = drugXA["data"].to_numpy()

    crossvalidate(X, rank=3)

    _, _, _, _ = parafac2_nd(X, rank=4, verbose=True) 

def test_pf2_r2x():
    """Compare R2X values to tensorly implementation"""
    X = random_parafac2([(10, 30)] * 5, rank=3, full=True, random_state=1)
    X = np.stack(X, axis=0)
    
    norm_tensor = tl.norm(X) ** 2

    weights, factors, projs, r2x = parafac2_nd(X, rank=3)
    
    err = _parafac2_reconstruction_error(X, (weights, factors, projs)) ** 2
    err = 1 - tl.to_numpy((err / norm_tensor))
    
    np.testing.assert_allclose(r2x, err, rtol=1e-3)