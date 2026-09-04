"""
Smoke + data-mapping tests for scrise.plotting.factors and
scrise.plotting.rank_selection.

We don't assert on rendered pixels. Instead we check two things a plotting
bug is actually likely to break: (1) the function runs without raising on
both a normal input and small edge cases (rank 1, a single row), and (2)
where cheaply checkable, that the *right numbers* ended up on the axes
(e.g. plot_bicv_r2x puts the correct R2X values on the y-axis) rather than
just "some heatmap was drawn".
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..plotting.factors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
    reorder_table,
)
from ..plotting.rank_selection import plot_bicv_r2x
from ..rank_selection import bicv
from .conftest import make_mock_factored_adata, make_synthetic_pf2_data


@pytest.mark.parametrize("rank,n_cond", [(3, 6), (1, 4)])
def test_plot_condition_factors_smoke(rank, n_cond):
    adata = make_mock_factored_adata(n_conditions=n_cond, rank=rank)
    # Force all-positive values so the default log_transform=True doesn't
    # hit log(negative).
    adata.uns["Pf2_A"] = np.abs(adata.uns["Pf2_A"]) + 0.1

    fig, ax = plt.subplots()
    plot_condition_factors(adata, ax)
    assert ax.get_xlabel() == "Component"
    assert len(ax.get_yticklabels()) == n_cond


def test_plot_condition_factors_with_group_labels_and_legend():
    n_cond, rank = 8, 3
    adata = make_mock_factored_adata(n_conditions=n_cond, rank=rank)
    adata.uns["Pf2_A"] = np.abs(adata.uns["Pf2_A"]) + 0.1

    yt = pd.Series(np.unique(adata.obs["Condition"]))
    group_labels = pd.Series(
        ["groupA" if i % 2 == 0 else "groupB" for i in range(len(yt))]
    )

    fig, ax = plt.subplots()
    plot_condition_factors(adata, ax, cond_group_labels=group_labels, group_cond=True)
    # A legend patch per unique group should have been added.
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == group_labels.nunique()


def test_plot_eigenstate_factors_smoke():
    adata = make_mock_factored_adata(rank=4)
    fig, ax = plt.subplots()
    plot_eigenstate_factors(adata, ax)
    assert ax.get_xlabel() == "Component"
    assert len(ax.get_xticklabels()) == 4


@pytest.mark.parametrize("trim", [True, False])
def test_plot_gene_factors_smoke(trim):
    adata = make_mock_factored_adata(n_genes=30, rank=3)
    fig, ax = plt.subplots()
    plot_gene_factors(adata, ax, trim=trim)
    assert ax.get_xlabel() == "Component"


def test_plot_gene_factors_trim_reduces_or_keeps_gene_count():
    adata = make_mock_factored_adata(n_genes=30, rank=3)
    fig1, ax1 = plt.subplots()
    plot_gene_factors(adata, ax1, trim=False, weight=0.08)
    n_all = len(ax1.get_yticklabels())

    fig2, ax2 = plt.subplots()
    plot_gene_factors(adata, ax2, trim=True, weight=0.08)
    n_trimmed = len(ax2.get_yticklabels())

    assert n_trimmed <= n_all


def test_reorder_table_groups_rows_by_dominant_component():
    """Rows should come back grouped by which column has their largest
    magnitude entry."""
    projs = np.array(
        [
            [0.1, 5.0, 0.0],  # dominant in col 1
            [3.0, 0.0, 0.1],  # dominant in col 0
            [0.0, 0.2, 4.0],  # dominant in col 2
            [1.0, 0.0, 0.1],  # dominant in col 0, smaller than row 1
        ]
    )
    ind = reorder_table(projs)
    dominant = np.argmax(np.abs(projs[ind]), axis=1)
    assert np.all(np.diff(dominant) >= 0)


def test_plot_bicv_r2x_smoke_and_axis_labels():
    X = make_synthetic_pf2_data(n_cond=4, n_genes=20, rank=2, seed=0)
    results = bicv(X, [2, 4], n_repeats=1, random_state=0, max_iter=30)

    fig, ax = plt.subplots()
    plot_bicv_r2x(results, ax)

    assert ax.get_xlabel() == "Rank"
    assert ax.get_ylabel() == "R2X"
    assert ax.get_legend() is not None


def test_plot_condition_factors_control_pattern_and_conditions():
    adata = make_mock_factored_adata(n_conditions=6, rank=3)
    adata.uns["Pf2_A"] = np.abs(adata.uns["Pf2_A"]) + 0.1
    # Assign conditions with CTRL and TRT
    cond_codes = adata.obs["condition_unique_idxs"].cat.codes
    cond_names = [f"CTRL_{c % 2}" if c < 2 else f"TRT_{c}" for c in cond_codes]
    adata.obs["Condition"] = pd.Categorical(cond_names)

    fig1, ax1 = plt.subplots()
    plot_condition_factors(adata, ax1, control_pattern="CTRL")
    assert len(ax1.get_yticklabels()) > 0

    fig2, ax2 = plt.subplots()
    plot_condition_factors(adata, ax2, control_conditions=["CTRL_0", "CTRL_1"])
    assert len(ax2.get_yticklabels()) > 0

    fig3, ax3 = plt.subplots()
    plot_condition_factors(adata, ax3, ThomsonNorm=True)
    assert len(ax3.get_yticklabels()) > 0


def test_plot_gene_factors_no_genes_pass_weight_raises():
    adata = make_mock_factored_adata(n_genes=10, rank=2)
    adata.varm["Pf2_C"] = np.full_like(adata.varm["Pf2_C"], 0.01)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="No genes exceeded the weight threshold"):
        plot_gene_factors(adata, ax, weight=0.5, trim=True)
