"""
Test the cross validation accuracy.
"""

import pytest
import numpy as np
from ..imports import import_thomson, import_lupus, import_citeseq


@pytest.mark.parametrize("import_func", [import_thomson, import_lupus, import_citeseq])
def test_imports(import_func):
    """Test import functions."""
    X = import_func()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32
