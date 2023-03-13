"""
Test the cross validation accuracy.
"""
import numpy as np
from tensorly.random import random_parafac2
from ..crossVal import crossvalidate


def test_crossval():
    """Test for correctness of cross validation."""
    rank = 3
    X = random_parafac2([(10, 5000)] * 30, rank=rank, full=True, random_state=1)
    X = np.stack(X, axis=0)

    err = crossvalidate(X, rank=rank, trainPerc=0.75)
    print(err)

    assert err > 0.85
