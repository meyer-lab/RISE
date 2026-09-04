"""
Smoke + data-shaping tests for scrise.plotting.general.

Several functions here (cell_count_perc_df, avegene_per_status, ...) are
really data-reshaping helpers that happen to feed a plot; those are tested
on their returned DataFrame contents directly rather than the plot.
"""

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..plotting.general import (
    avegene_per_status,
    cell_count_perc_df,
    cell_count_perc_lupus_df,
    gene_plot_cells,
    plot_avegene_per_celltype,
    plot_cell_gene_corr,
    plot_r2x,
    rotate_xaxis,
    rotate_yaxis,
)
from .conftest import make_synthetic_pf2_data


def _two_gene_adata(n_cells=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, 2)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "hue": pd.Categorical([f"grp_{i % 3}" for i in range(n_cells)]),
            "Cell Type": pd.Categorical([f"type_{i % 2}" for i in range(n_cells)]),
        }
    )
    var = pd.DataFrame({"means": [0.0, 0.0]}, index=["gene_a", "gene_b"])
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_rotate_xaxis_and_yaxis_set_tick_rotation():
    fig, ax = plt.subplots()
    rotate_xaxis(ax, rotation=45)
    rotate_yaxis(ax, rotation=30)
    # matplotlib doesn't expose rotation as a simple getter pre-draw; the
    # smoke check is that these don't raise and the axes object still works.
    ax.figure.canvas.draw()


def test_cell_count_perc_df_percentages_sum_to_100_per_condition():
    n_cells = 60
    obs = pd.DataFrame(
        {
            "Cell Type": np.random.choice(["A", "B", "C"], size=n_cells),
            "Condition": np.random.choice(["cond_0", "cond_1"], size=n_cells),
        }
    )
    X = anndata.AnnData(X=np.zeros((n_cells, 2), dtype=np.float32), obs=obs)

    df = cell_count_perc_df(X)
    totals = df.groupby("Condition")["Cell Type Percentage"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 100.0, rtol=1e-6)
    assert df["Cell Count"].sum() == n_cells


def test_cell_count_perc_lupus_df_attaches_metadata_columns():
    n_cells = 40
    obs = pd.DataFrame(
        {
            "Cell Type": np.random.choice(["A", "B"], size=n_cells),
            "Condition": np.repeat(["cond_0", "cond_1"], n_cells // 2),
            "SLE_status": np.repeat(["healthy", "SLE"], n_cells // 2),
            "Processing_Cohort": np.repeat([1, 2], n_cells // 2),
            "condition_unique_idxs": np.repeat([0, 1], n_cells // 2),
        }
    )
    X = anndata.AnnData(X=np.zeros((n_cells, 2), dtype=np.float32), obs=obs)

    df = cell_count_perc_lupus_df(X)
    assert {"SLE_status", "Processing_Cohort", "condition_unique_idxs"}.issubset(
        df.columns
    )
    cond0_status = df.loc[df["Condition"] == "cond_0", "SLE_status"].unique()
    assert list(cond0_status) == ["healthy"]


def test_avegene_per_status_returns_expected_columns():
    n_cells = 30
    X = _two_gene_adata(n_cells)
    X.obs["SLE_status"] = np.random.choice(["healthy", "SLE"], size=n_cells)
    X.obs["Condition"] = np.random.choice(["cond_0", "cond_1"], size=n_cells)

    df = avegene_per_status(X[:, "gene_a"], "gene_a")
    assert {
        "Status",
        "Cell Type",
        "Gene",
        "Condition",
        "Average Gene Expression",
    } <= set(df.columns)
    assert set(df["Gene"]) == {"gene_a"}


def test_plot_avegene_per_celltype_smoke():
    n_cells = 30
    X = _two_gene_adata(n_cells)
    X.obs["Condition"] = np.random.choice(["cond_0", "cond_1"], size=n_cells)

    fig, ax = plt.subplots()
    plot_avegene_per_celltype(X, ["gene_a", "gene_b"], ax)


def test_gene_plot_cells_smoke_and_shape_assertion():
    X = _two_gene_adata(20)
    fig, ax = plt.subplots()
    gene_plot_cells(X, hue="hue", ax=ax)

    # gene_plot_cells asserts exactly 2 genes/columns are present.
    X_three_genes = anndata.AnnData(
        X=np.zeros((5, 3), dtype=np.float32),
        var=pd.DataFrame({"means": [0.0, 0.0, 0.0]}, index=["a", "b", "c"]),
        obs=pd.DataFrame({"hue": ["x"] * 5, "Cell Type": ["t"] * 5}),
    )
    fig2, ax2 = plt.subplots()
    with pytest.raises(AssertionError):
        gene_plot_cells(X_three_genes, hue="hue", ax=ax2)


def test_plot_cell_gene_corr_smoke_with_missing_pivot_columns():
    """When the requested (gene, cell-type) combination isn't present after
    pivoting, plot_cell_gene_corr should fall back to an empty frame rather
    than raising a KeyError."""
    X = _two_gene_adata(20)
    fig, ax = plt.subplots()
    plot_cell_gene_corr(
        X, hue="hue", cells=["type_0", "type_1"], ax=ax, unique=["grp_0"]
    )


def test_plot_r2x_smoke_and_axes_labels():
    X = make_synthetic_pf2_data(n_cond=4, n_genes=15, rank=2, seed=0)
    fig, ax = plt.subplots()
    plot_r2x(X, np.array([1, 2, 3]), ax, compress=None)

    assert ax.get_xlabel() == "Number of Components"
    assert ax.get_ylabel() == "Variance Explained"
    assert ax.get_legend() is not None
