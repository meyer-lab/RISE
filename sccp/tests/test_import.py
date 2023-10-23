"""
Test the cross validation accuracy.
"""
import numpy as np
from ..imports import import_thomson, import_lupus, import_citeseq


def test_Thomson():
    """Test for correctness of cross validation."""
    X = import_thomson()


def test_Lupus():
    """Test for correctness of cross validation."""
    X = import_lupus()


def test_CITEseq():
    """Test for correctness of cross validation."""
    X = import_citeseq()
