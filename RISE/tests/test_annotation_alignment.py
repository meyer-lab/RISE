"""
Unit tests for cell-type alignment scoring and visualization.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from RISE import (
    CellTypeAlignmentResults,
    ComponentAlignmentResult,
    cell_type_alignment,
    compute_tau,
    score_cell_type_alignment,
)
from RISE.annotation_alignment import (
    compute_auroc_per_cell_type,
    compute_eta_squared,
    compute_kruskal_epsilon_squared,
)
from RISE.plotting import plot_cell_type_alignment


@pytest.fixture
def synthetic_alignment_data():
    """Create synthetic AnnData with 3 distinct components:
    - Component 1: High only in TypeA (private -> high tau, significant for TypeA only)
    - Component 2: High in TypeA and TypeB (moderate tau, both significant)
    - Component 3: Random noise (tau and eta^2 near 0, no significant cell types)
    """
    rng = np.random.default_rng(42)
    n_cells = 600
    types = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE"]
    cell_types = pd.Series(rng.choice(types, size=n_cells), name="cell_type")

    # Comp 1: private to TypeA
    c1 = rng.normal(0.0, 0.5, size=n_cells)
    c1[cell_types == "TypeA"] += 5.0

    # Comp 2: high in TypeA and TypeB
    c2 = rng.normal(0.0, 0.5, size=n_cells)
    c2[cell_types == "TypeA"] += 5.0
    c2[cell_types == "TypeB"] += 5.0

    # Comp 3: random noise
    c3 = rng.normal(0.0, 0.5, size=n_cells)

    loadings = np.column_stack([c1, c2, c3]).astype(np.float32)

    adata = anndata.AnnData(
        obs=pd.DataFrame({"cell_type": cell_types}),
        obsm=cast(
            Mapping[str, Sequence[Any]],
            {"weighted_projections": loadings, "projections": loadings},
        ),
    )
    return adata, cell_types, loadings


def test_synthetic_three_components_expectations(synthetic_alignment_data):
    """Test the three core scenarios:
    1. One component high only in one cell type (high tau, single significant type).
    2. One component high in two of several types (moderate tau, both types significant).
    3. One component random noise unrelated to cell type (tau and eta^2 near 0, no significant types).
    """
    _, cell_types, loadings = synthetic_alignment_data

    # Test single-component scoring
    # Comp 1: High in TypeA only
    res1 = cell_type_alignment(
        loadings[:, 0], cell_types, n_permutations=500, random_state=42
    )
    assert isinstance(res1, ComponentAlignmentResult)
    assert res1.tau > 0.55
    assert res1.eta_squared > 0.7
    assert res1.significant_cell_types == ["TypeA"]
    assert res1.enrichment["TypeA"] > 0.95

    # Comp 2: High in TypeA and TypeB
    res2 = cell_type_alignment(
        loadings[:, 1], cell_types, n_permutations=500, random_state=42
    )
    assert 0.4 < res2.tau < 0.65
    assert res2.eta_squared > 0.7
    assert set(res2.significant_cell_types) == {"TypeA", "TypeB"}
    assert res2.enrichment["TypeA"] > 0.8
    assert res2.enrichment["TypeB"] > 0.8

    # Comp 3: Random noise
    res3 = cell_type_alignment(
        loadings[:, 2], cell_types, n_permutations=500, random_state=42
    )
    assert res3.tau < 0.15
    assert res3.eta_squared < 0.05
    assert len(res3.significant_cell_types) == 0


def test_score_cell_type_alignment_anndata(synthetic_alignment_data):
    """Test score_cell_type_alignment across all components from AnnData."""
    adata, _, _ = synthetic_alignment_data

    res = score_cell_type_alignment(
        adata,
        cell_types="cell_type",
        projection_key="weighted_projections",
        n_permutations=500,
        random_state=42,
    )

    assert isinstance(res, CellTypeAlignmentResults)
    assert res.enrichment.shape == (3, 5)
    assert res.q_values.shape == (3, 5)
    assert len(res.results) == 3

    # Check component 1
    assert res.significant_cell_types[1] == ["TypeA"]
    assert res.tau.loc[1] > 0.55

    # Check component 2
    assert set(res.significant_cell_types[2]) == {"TypeA", "TypeB"}

    # Check component 3
    assert len(res.significant_cell_types[3]) == 0
    assert res.tau.loc[3] < 0.15
    assert res.eta_squared.loc[3] < 0.05

    # Summary table
    summary_df = res.summary()
    assert summary_df.shape == (3, 5)
    assert "top_cell_type" in summary_df.columns
    assert "significant_cell_types" in summary_df.columns
    assert summary_df.loc[1, "top_cell_type"] == "TypeA"


def test_signed_loadings():
    """Test signed=True handles signed loadings by taking absolute values."""
    rng = np.random.default_rng(42)
    n_cells = 400
    types = ["Type1", "Type2", "Type3"]
    cell_types = pd.Series(rng.choice(types, size=n_cells))

    # Loading with large negative values in Type1
    loadings = rng.normal(0, 0.2, size=n_cells)
    loadings[cell_types == "Type1"] = -5.0

    res_unsigned = cell_type_alignment(
        loadings, cell_types, signed=False, n_permutations=200, random_state=42
    )
    assert (
        res_unsigned.enrichment["Type1"] < 0.1
    )  # Depleted because values are strongly negative

    res_signed = cell_type_alignment(
        loadings, cell_types, signed=True, n_permutations=200, random_state=42
    )
    assert res_signed.enrichment["Type1"] > 0.9  # Enriched in magnitude
    assert res_signed.significant_cell_types == ["Type1"]


def test_asymptotic_mann_whitney():
    """Test n_permutations=0 runs asymptotic Mann-Whitney."""
    rng = np.random.default_rng(42)
    n_cells = 200
    types = ["Type1", "Type2"]
    cell_types = pd.Series(rng.choice(types, size=n_cells))

    loadings = rng.normal(0, 1, size=n_cells)
    loadings[cell_types == "Type1"] += 3.0

    res = cell_type_alignment(loadings, cell_types, n_permutations=0)
    assert res.p_values["Type1"] < 1e-4
    assert res.significant_cell_types == ["Type1"]


def test_to_dict_and_helpers():
    """Test to_dict, compute_tau, and effect size helpers."""
    # Test compute_tau
    assert compute_tau(np.array([1.0, 0.0, 0.0]), baseline=0.0) == 1.0
    assert compute_tau(np.array([0.5, 0.5, 0.5]), baseline=0.0) == 0.0
    assert compute_tau(np.array([0.5])) == 0.0

    # Test eta_squared and epsilon_squared with constant loading
    const = np.ones(50)
    codes = np.zeros(50, dtype=int)
    assert compute_eta_squared(const, codes, 1) == 0.0
    assert compute_kruskal_epsilon_squared(const, codes, 1) == 0.0

    # Test single-cell / single-type AUROC
    assert compute_auroc_per_cell_type(np.array([]), np.array([]), 0).size == 0


def test_plotting_functions(synthetic_alignment_data):
    """Test plot_cell_type_alignment visualization."""
    adata, _, loadings = synthetic_alignment_data

    # 1. From AnnData (default creates fig with 2 subplots)
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    out_axes = plot_cell_type_alignment(
        adata, ax=axes, n_permutations=100, random_state=42
    )
    assert isinstance(out_axes, tuple)
    assert len(out_axes) == 2
    plt.close("all")

    # 2. Single axes (no metrics subplot)
    _, ax = plt.subplots(figsize=(6, 4))
    out_ax = plot_cell_type_alignment(
        adata, ax=ax, show_metrics=False, n_permutations=100, random_state=42
    )
    assert out_ax == ax
    plt.close("all")

    # 3. From precomputed results
    res = score_cell_type_alignment(adata, n_permutations=100, random_state=42)
    out_axes2 = plot_cell_type_alignment(res, show_metrics=True)
    assert isinstance(out_axes2, (tuple, list))
    plt.close("all")

    # 4. From raw DataFrame
    df = pd.DataFrame(loadings[:5, :3], columns=["T1", "T2", "T3"])
    out_ax3 = plot_cell_type_alignment(df, show_metrics=False)
    assert out_ax3 is not None
    plt.close("all")


def test_error_handling(synthetic_alignment_data):
    """Test error handling for mismatched dimensions or invalid keys."""
    adata, cell_types, loadings = synthetic_alignment_data

    # Dimension mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        cell_type_alignment(loadings[:, 0], cell_types.iloc[:10])

    # Missing projection key
    with pytest.raises(KeyError, match="Could not find"):
        score_cell_type_alignment(adata, projection_key="non_existent_key")

    # Missing cell type column
    with pytest.raises(KeyError, match="Column 'non_existent_col' not found"):
        score_cell_type_alignment(adata, cell_types="non_existent_col")
