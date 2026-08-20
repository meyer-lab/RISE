"""
Test bi-cross-validation rank selection.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from ..rank_selection import BiCVOptimizationResult, bicv, optimize_rank


def _make_test_data(
    n_cond: int = 5, n_genes: int = 40, true_rank: int = 3, seed: int = 0
) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(true_rank, true_rank))
    C = rng.normal(size=(n_genes, true_rank))
    A = rng.normal(size=(n_cond, true_rank))

    X_list = []
    cond_idx = []
    for i in range(n_cond):
        n_cells = int(rng.integers(60, 90))
        Z = rng.normal(size=(n_cells, true_rank))
        signal = (Z @ B) * A[i] @ C.T
        noise = rng.normal(scale=0.2, size=signal.shape)
        X_list.append(signal + noise)
        cond_idx += [i] * n_cells

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    adata = anndata.AnnData(X=X)
    adata.obs["condition_unique_idxs"] = pd.Categorical(cond_idx)
    adata.var["means"] = np.zeros(n_genes)
    return adata


def test_bicv_shape_and_range():
    """bicv() returns a long-form DataFrame with the expected columns and
    R2X values in a sane range."""
    X = _make_test_data()
    ranks = [2, 4, 6]
    n_repeats = 2

    results = bicv(X, ranks, n_repeats=n_repeats, random_state=0, max_iter=50)

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {"Rank", "Repeat", "Metric", "R2X"}
    assert set(results["Metric"]) == {"Fit R2X", "BiCV R2X"}
    assert set(results["Rank"]) == set(ranks)

    fit_rows = results[results["Metric"] == "Fit R2X"]
    assert len(fit_rows) == len(ranks)

    bicv_rows = results[results["Metric"] == "BiCV R2X"]
    assert len(bicv_rows) == len(ranks) * n_repeats

    # Fit R2X should increase monotonically with rank (in-sample fit).
    fit_by_rank = fit_rows.set_index("Rank")["R2X"].sort_index()
    assert np.all(np.diff(fit_by_rank.to_numpy()) >= -1e-6)

    # R2X values should be finite and not wildly out of range.
    assert np.all(np.isfinite(results["R2X"]))
    assert np.all(results["R2X"] < 1.0 + 1e-6)


def test_bicv_invalid_arguments():
    X = _make_test_data()

    with pytest.raises(ValueError):
        bicv(X, [5], held_out_cell_frac=1.5)

    with pytest.raises(ValueError):
        bicv(X, [5], n_repeats=0)

    with pytest.raises(ValueError):
        # Rank too large to be feasible given default held-out fractions.
        bicv(X, [1000])


def test_optimize_rank_finds_reasonable_result():
    """optimize_rank() returns a best rank within bounds and a usable
    history/GP for plotting."""
    X = _make_test_data()

    result = optimize_rank(
        X,
        rank_bounds=(2, 10),
        n_repeats=2,
        n_calls=6,
        n_initial_points=3,
        random_state=0,
        max_iter=50,
    )

    assert isinstance(result, BiCVOptimizationResult)
    assert 2 <= result.best_rank <= 10
    assert np.isfinite(result.best_r2x)
    assert len(result.history) == 6
    assert set(result.history.columns) == {"Rank", "R2X", "R2X_std"}

    # The GP surrogate should be usable for prediction over the search range.
    grid = np.arange(2, 11).reshape(-1, 1).astype(float)
    mu, sigma = result.gp.predict(grid, return_std=True)
    assert mu.shape == (9,)
    assert np.all(sigma >= 0)


def test_optimize_rank_invalid_arguments():
    X = _make_test_data()

    with pytest.raises(ValueError):
        optimize_rank(X, rank_bounds=(10, 2))

    with pytest.raises(ValueError):
        optimize_rank(X, rank_bounds=(2, 10), n_calls=2, n_initial_points=5)
