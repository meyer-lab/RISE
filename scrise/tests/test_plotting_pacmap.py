"""
Smoke tests for scrise.plotting.pacmap.

PaCMAP itself is not exercised here -- these functions only ever consume a
precomputed 2D embedding (X.obsm["X_pf2_PaCMAP"]), so we supply one
directly rather than paying for an actual PaCMAP fit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..plotting.pacmap import (
    _get_canvas,
    _to_hex,
    assign_labels,
    plot_gene_pacmap,
    plot_labels_pacmap,
    plot_wp_pacmap,
)
from .conftest import make_mock_factored_adata


def test_get_canvas_bounds_pad_outward_from_data_range():
    points = np.array([[0.0, 0.0], [10.0, 20.0]])
    canvas = _get_canvas(points)
    assert canvas.x_range[0] <= 0.0
    assert canvas.x_range[1] >= 10.0
    assert canvas.y_range[0] <= 0.0
    assert canvas.y_range[1] >= 20.0


def test_to_hex_returns_hex_strings():
    colors = _to_hex(plt.get_cmap("tab10")(np.linspace(0, 1, 4)))
    assert len(colors) == 4
    assert all(c.startswith("#") for c in colors)


def test_assign_labels_sets_pacmap_axis_labels():
    fig, ax = plt.subplots()
    out = assign_labels(ax)
    assert out is ax
    assert ax.get_xlabel() == "PaCMAP1"
    assert ax.get_ylabel() == "PaCMAP2"
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []


def test_plot_gene_pacmap_smoke():
    adata = make_mock_factored_adata(n_cells=50, n_genes=10, with_embedding=True)
    gene = adata.var_names[0]
    fig, ax = plt.subplots()
    plot_gene_pacmap(gene, adata, ax)
    assert ax.get_title() == gene


@pytest.mark.parametrize("cmp", [1, 3])
def test_plot_wp_pacmap_smoke(cmp):
    adata = make_mock_factored_adata(n_cells=50, rank=3, with_embedding=True)
    fig, ax = plt.subplots()
    plot_wp_pacmap(adata, cmp=cmp, ax=ax)
    assert ax.get_title() == f"Cmp. {cmp}"


def test_plot_labels_pacmap_smoke_and_legend_matches_categories():
    adata = make_mock_factored_adata(n_cells=60, with_embedding=True)
    fig, ax = plt.subplots()
    plot_labels_pacmap(adata, labelType="Cell Type", ax=ax)

    legend = ax.get_legend()
    assert legend is not None
    n_categories = adata.obs["Cell Type"].nunique()
    assert len(legend.get_texts()) == n_categories


def test_plot_labels_pacmap_condition_filter_collapses_to_other():
    adata = make_mock_factored_adata(n_cells=60, with_embedding=True)
    present = adata.obs["Cell Type"].unique().tolist()
    keep = [present[0]]

    fig, ax = plt.subplots()
    plot_labels_pacmap(adata, labelType="Cell Type", ax=ax, condition=keep)

    legend = ax.get_legend()
    assert legend is not None
    legend_labels = {t.get_text() for t in legend.get_texts()}
    # Everything not in `keep` should be collapsed into a single "Other"
    # category, so at most len(keep) + 1 categories should appear.
    assert legend_labels <= set(keep) | {"Other"}
