"""
Test the cross validation accuracy.
"""

import numpy as np
import pytest

from ..imports import (
    import_thomson,
)


@pytest.mark.parametrize(
    "import_func",
    [
        import_thomson,
    ],
)
def test_imports(import_func):
    """Test import functions."""
    X = import_func()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32
