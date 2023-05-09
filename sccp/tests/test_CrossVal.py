"""
Test the cross validation accuracy.
"""
import numpy as np
from tensorly.random import random_parafac2
from ..crossVal import crossvalidate, crossvalidate_PCA


def test_crossval():
    """Test for correctness of cross validation."""
    rank = 5
    rng = np.random.default_rng(2)

    X = random_parafac2([(100, 200)] * 5, rank=rank, full=True, random_state=2)

    pca_err = crossvalidate_PCA(np.concatenate(X), rank=rank, trainPerc=0.8, random_state=rng)
    assert pca_err[-1] > 0.95

    err = crossvalidate(X, rank=rank, trainPerc=0.8, random_state=rng)
    assert err > 0.98
