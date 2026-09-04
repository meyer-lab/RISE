"""
Test the parafac2 method.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.imports import import_thomson

from ..factorization import pf2, rise_pca_r2x


def test_factor_thomson_R2X():
    """Import and factor Thomson.
    Check that the factorization process is reproducible."""
    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    r2x_rise, r2x_pca = rise_pca_r2x(X, np.arange(1, 4))
    print(r2x_pca)
    print(r2x_rise)

    assert np.all(r2x_rise > np.array([0.0001, 0.0002, 0.0003]))


def test_factor_thomson_mlx_backend():
    """Run the factorization on the MLX (Apple GPU) backend.

    Skipped unless MLX is installed, which is only the case on the
    macOS/arm64 CI runner where the `gpu` extra is installed.
    """
    pytest.importorskip("mlx.core")

    X = import_thomson()[::10, :500].copy()
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])

    res = pf2(X, 5, doEmbedding=False, tolerance=1e-6, backend="mlx")

    assert np.all(np.isfinite(np.asarray(res.varm["Pf2_C"])))


def test_pf2_condition_key():
    import anndata

    from .conftest import make_synthetic_pf2_data

    orig = make_synthetic_pf2_data(n_cond=3, n_genes=15, rank=2, seed=42)
    obs = pd.DataFrame({"custom_condition": orig.obs["Condition"].to_numpy()})
    X = anndata.AnnData(X=orig.X, obs=obs, var=pd.DataFrame(index=orig.var_names))

    with pytest.raises(KeyError, match="condition_unique_idxs"):
        pf2(X, 2, doEmbedding=False, max_iter=5, compress=None)

    res = pf2(
        X,
        2,
        condition_key="custom_condition",
        doEmbedding=False,
        max_iter=5,
        compress=None,
    )
    assert "condition_unique_idxs" in res.obs
    assert "Pf2_A" in res.uns


def test_pf2_adata_alias():
    from .conftest import make_synthetic_pf2_data

    X = make_synthetic_pf2_data(n_cond=3, n_genes=15, rank=2, seed=42)
    res = pf2(adata=X, rank=2, doEmbedding=False, max_iter=5, compress=None)
    assert "Pf2_A" in res.uns


def test_root_api_exports():
    import scrise
    from scrise import __version__, bicv, plotting, prepare_dataset

    assert callable(bicv)
    assert callable(prepare_dataset)
    assert hasattr(plotting, "plot_condition_factors")
    assert hasattr(plotting, "plot_gene_factors")
    assert __version__ == "1.2.0"
    assert "bicv" in scrise.__all__
    assert "prepare_dataset" in scrise.__all__
    assert "plotting" in scrise.__all__
    assert "__version__" in scrise.__all__
