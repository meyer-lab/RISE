"""
Test the cross validation accuracy.
"""
import pytest
from ..imports import import_thomson, import_lupus, import_citeseq


def test_Thomson():
    """Test for correctness of cross validation."""
    X = import_thomson()
    print()
    print(f"Data shape: {X.shape}")
    print(f"Data non-zeros: {X.X.nnz}")


@pytest.mark.skip("The lupus dataset uses too much memory for now.")
def test_Lupus():
    """Test for correctness of cross validation."""
    X = import_lupus()


def test_CITEseq():
    """Test for correctness of cross validation."""
    X = import_citeseq()
    print()
    print(f"Data shape: {X.shape}")
    print(f"Data non-zeros: {X.X.nnz}")
