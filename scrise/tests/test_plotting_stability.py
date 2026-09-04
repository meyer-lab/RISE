"""
Tests for scrise.plotting.stability (Factor Match Score and its plots).

calculateFMS and resample are pure/cheap enough to test directly and
precisely. plot_fms_diff_ranks / plot_fms_percent_drop each refit pf2()
several times, so they're only smoke-tested, on a tiny synthetic dataset
and the smallest run/rank counts that still exercise the real code path.
"""

import anndata
import matplotlib.pyplot as plt
import numpy as np

from ..plotting.stability import (
    calculateFMS,
    plot_fms_diff_ranks,
    plot_fms_percent_drop,
    resample,
)
from .conftest import make_synthetic_pf2_data


def _mock_factors_adata(A, B, C, weights):
    return anndata.AnnData(
        X=np.zeros((2, C.shape[0]), dtype=np.float32),
        uns={"Pf2_A": A, "Pf2_B": B, "Pf2_weights": weights},
        varm={"Pf2_C": C},
    )


def test_calculate_fms_identical_decompositions_score_near_one():
    rng = np.random.default_rng(0)
    n_cond, n_genes, rank = 5, 20, 3
    A = rng.normal(size=(n_cond, rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(n_genes, rank))
    weights = rng.random(rank)

    X = _mock_factors_adata(A.copy(), B.copy(), C.copy(), weights.copy())
    Y = _mock_factors_adata(A.copy(), B.copy(), C.copy(), weights.copy())

    assert np.isclose(calculateFMS(X, Y), 1.0, atol=1e-6)


def test_calculate_fms_permuted_components_still_scores_near_one():
    """FMS should be invariant to a permutation of component order -- it's
    matching components, not comparing them positionally."""
    rng = np.random.default_rng(1)
    n_cond, n_genes, rank = 5, 20, 3
    A = rng.normal(size=(n_cond, rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(n_genes, rank))
    weights = rng.random(rank)

    perm = np.array([2, 0, 1])
    X = _mock_factors_adata(A.copy(), B.copy(), C.copy(), weights.copy())
    Y = _mock_factors_adata(
        A[:, perm].copy(), B[:, perm].copy(), C[:, perm].copy(), weights[perm].copy()
    )

    assert np.isclose(calculateFMS(X, Y), 1.0, atol=1e-6)


def test_calculate_fms_unrelated_decompositions_scores_low():
    rng = np.random.default_rng(2)
    n_cond, n_genes, rank = 5, 20, 3
    A1, B1, C1 = (
        rng.normal(size=(n_cond, rank)),
        rng.normal(size=(rank, rank)),
        rng.normal(size=(n_genes, rank)),
    )
    A2, B2, C2 = (
        rng.normal(size=(n_cond, rank)),
        rng.normal(size=(rank, rank)),
        rng.normal(size=(n_genes, rank)),
    )
    weights = rng.random(rank)

    X = _mock_factors_adata(A1, B1, C1, weights.copy())
    Y = _mock_factors_adata(A2, B2, C2, weights.copy())

    assert calculateFMS(X, Y) < 0.5


def test_resample_preserves_shape_and_samples_with_replacement():
    n_cells, n_genes = 50, 10
    X = anndata.AnnData(
        X=np.arange(n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes)
    )
    np.random.seed(0)
    resampled = resample(X)
    assert resampled.shape == X.shape
    # With replacement over 50 draws, near-certain to hit a duplicate row.
    assert len(np.unique(np.asarray(resampled.X)[:, 0])) < n_cells


def test_plot_fms_diff_ranks_smoke():
    X = make_synthetic_pf2_data(n_cond=4, n_genes=15, rank=2, seed=0)
    fig, ax = plt.subplots()
    plot_fms_diff_ranks(X, ax, ranksList=[2], runs=1, compress=None)

    assert ax.get_xlabel() == "Component"
    assert ax.get_ylabel() == "FMS"
    assert ax.get_ylim() == (0.0, 1.0)


def test_plot_fms_percent_drop_smoke():
    X = make_synthetic_pf2_data(
        n_cond=3, n_genes=10, rank=2, seed=0, cells_per_cond=(20, 30)
    )
    fig, ax = plt.subplots()
    plot_fms_percent_drop(X, ax, percentList=np.array([0, 10]), runs=1, rank=2)
    assert ax.get_ylabel() == "FMS"
    assert ax.get_ylim() == (0.0, 1.0)
