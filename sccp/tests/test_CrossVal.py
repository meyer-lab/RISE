"""
Test the cross validation accuracy.
"""
import tensorly as tl
import numpy as np
from tensorly.random import random_parafac2
from ..crossVal import crossvalidate, crossvalidate_PCA


def test_crossval():
    """Test for correctness of cross validation."""
    rank = 3
    X = random_parafac2([(50, 50)] * 8, rank=rank, full=True)

    pca_err = crossvalidate_PCA(np.concatenate(X, axis=0), rank=rank, trainPerc=0.8)
    assert pca_err > 0.95

    err = crossvalidate(X, rank=rank, trainPerc=0.8)
    assert err > 0.95
