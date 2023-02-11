"""
Test the data import.
"""
import numpy as np
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from tensorly.metrics import correlation_index
from ..parafac2 import parafac2_nd
from ..imports.scRNA import ThompsonXA_SCGenes


def test_n_way():
    """Compare the PARAFAC2 results N-way to TensorLy in 3D."""
    X = random_parafac2([(10, 30)] * 5, rank=3, full=True, random_state=1)
    X = np.stack(X, axis=0)
    rcpStack = np.stack([X, X, X], axis=0)

    _, factors, projections = parafac2(X, rank=3, normalize_factors=True, init="svd")
    _, facStack, projStack = parafac2_nd(rcpStack, rank=3)

    # More similar is closer to 0 with corrIndex
    assert correlation_index(factors, facStack[1::], method="min_score") < 0.1

    # Should just be a flat set of factors
    np.testing.assert_allclose(facStack[0], np.full((3, 3), 1.0 / np.sqrt(3.0)))

    # Compare projection matrices, too
    assert (
        correlation_index(
            list(projections), list(projStack[0, :, :, :]), method="min_score"
        )
        < 0.1
    )


def test_pf2_speed():
    """Compare run time for different SVD initialization """
    drugXA, _ = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    
    _, _, _ = parafac2_nd(drugXA.to_numpy(), rank=6, svd="randomized_svd", n_iter_max=10, verbose=True)
