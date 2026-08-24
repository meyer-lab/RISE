"""
Test the cross validation accuracy.
"""

import numpy as np
import pytest

from analysis.imports import import_lupus, import_thomson, import_thomson_factors


@pytest.mark.parametrize(
    "import_func",
    [
        import_thomson,
        import_lupus,
    ],
)
def test_imports(import_func):
    """Test import functions."""
    X = import_func()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32


def test_import_thomson_factors():
    """Test import_thomson_factors function."""
    factors = import_thomson_factors(include_raw=False)
    assert factors.X is None
    assert "projections" in factors.obsm
    assert "weighted_projections" in factors.obsm
    assert "Pf2_C" in factors.varm
    assert "Pf2_A" in factors.uns

    factors_raw = import_thomson_factors(include_raw=True)
    assert factors_raw.X is not None
    assert factors_raw.X.dtype == np.float32
    assert factors_raw.shape == (29433, 12164)

