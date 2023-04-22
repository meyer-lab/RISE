"""
Test the cross validation accuracy.
"""
import numpy as np
from tensorly.random import random_parafac2
from ..crossVal import crossvalidate, crossvalidate_PCA


def test_crossval():
    """Test for correctness of cross validation."""
    rank = 5
    X = random_parafac2([(100, 200)] * 5, rank=rank, full=True)

    pca_err = crossvalidate_PCA(np.concatenate(X), rank=rank, trainPerc=0.8)
    assert pca_err[-1] > 0.95

    err = crossvalidate(X, rank=rank, trainPerc=0.8)
    assert err > 0.95
