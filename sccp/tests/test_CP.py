"""
Test the data import.
"""
import numpy as np
from tensorly.decomposition import parafac2
from tensorly.random import random_parafac2
from tensorly.metrics import correlation_index
from ..parafac2 import parafac2_nd
from ..imports.scRNA import ThompsonXA_SCGenes
from ..crossVal import crossvalidate


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
        correlation_index(
            list(projections), list(projStack), method="min_score"
        )
        < 0.1
    )


def test_pf2_speed():
    """Compare run time for different SVD initialization """
    drugXA, _ = ThompsonXA_SCGenes(saveXA=False, offset=1.0)

    crossvalidate(drugXA.to_numpy(), rank=3)
    
    _, _, _, _ = parafac2_nd(drugXA.to_numpy(), rank=4, verbose=True)
