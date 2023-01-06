"""
Test the data import.
"""
from ..parafac2 import parafac2
from tensorly.random import random_cp


def test_projection():
    """Stub test."""
    rcp = random_cp((10, 20, 30, 30), rank=3, full=True)
    _, factors, _ = parafac2(rcp, rank=3, n_iter_max=20, verbose=True, tol=1e-10)

