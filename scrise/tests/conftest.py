"""
Shared fixtures and synthetic-data factories for the scrise test suite.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import anndata
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib figures from accumulating across tests."""
    yield
    plt.close("all")


def make_synthetic_pf2_data(
    n_cond: int = 5,
    n_genes: int = 40,
    rank: int = 3,
    seed: int = 0,
    cells_per_cond: tuple[int, int] = (60, 90),
) -> anndata.AnnData:
    """Build a synthetic AnnData with a known low-rank PARAFAC2-like structure.

    Mirrors the generator originally written for ``test_rank_selection.py``
    so every test module shares one synthetic-data factory instead of
    hand-rolling its own. The data is a noisy realization of
    ``(Z_i @ B) * A[i] @ C.T`` per condition ``i``, which is exactly the
    structure PARAFAC2 assumes, so a real ``pf2()`` fit on this data
    recovers ``A``/``B``/``C`` up to the usual permutation/sign/scale
    ambiguity.
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(n_genes, rank))
    A = rng.normal(size=(n_cond, rank))

    X_list = []
    cond_idx = []
    for i in range(n_cond):
        n_cells = int(rng.integers(*cells_per_cond))
        Z = rng.normal(size=(n_cells, rank))
        signal = (Z @ B) * A[i] @ C.T
        noise = rng.normal(scale=0.2, size=signal.shape)
        X_list.append(signal + noise)
        cond_idx += [i] * n_cells

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    n_cells_total = X.shape[0]

    obs = pd.DataFrame(
        {
            "condition_unique_idxs": pd.Categorical(cond_idx),
            "Condition": pd.Categorical([f"cond_{i}" for i in cond_idx]),
        }
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{j}" for j in range(n_genes)]},
        index=[f"gene_{j}" for j in range(n_genes)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var["means"] = np.zeros(n_genes)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells_total)]
    return adata


def make_mock_factored_adata(
    n_cells: int = 40,
    n_genes: int = 25,
    n_conditions: int = 6,
    rank: int = 3,
    seed: int = 0,
    with_embedding: bool = False,
) -> anndata.AnnData:
    """Build an AnnData already populated with (random, not fitted) RISE
    factors -- i.e. what ``pf2()`` would have attached -- for testing
    downstream consumers (reordering, plotting, export) without paying for
    an actual PARAFAC2 fit."""
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(n_conditions, rank)).astype(np.float32)
    C = rng.normal(size=(n_genes, rank)).astype(np.float32)
    B = rng.normal(size=(rank, rank)).astype(np.float32)
    weights = rng.random(rank).astype(np.float32)
    projections, _ = np.linalg.qr(rng.normal(size=(n_cells, rank)))
    projections = projections.astype(np.float32)

    cond_idx = [i % n_conditions for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            "Condition": pd.Categorical([f"cond_{i}" for i in cond_idx]),
            "condition_unique_idxs": pd.Categorical(cond_idx),
            "Cell Type": pd.Categorical([f"type_{i % 3}" for i in range(n_cells)]),
        }
    )
    var = pd.DataFrame(
        {"means": np.zeros(n_genes)},
        index=[f"gene_{j}" for j in range(n_genes)],
    )

    obsm = {"projections": projections, "weighted_projections": projections @ B}
    if with_embedding:
        obsm["X_pf2_PaCMAP"] = rng.normal(size=(n_cells, 2)).astype(np.float32)

    adata = anndata.AnnData(
        X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
        obs=obs,
        var=var,
        uns={"Pf2_A": A, "Pf2_B": B, "Pf2_weights": weights},
        varm=cast(Mapping[str, Sequence[Any]], {"Pf2_C": C}),
        obsm=cast(Mapping[str, Sequence[Any]], obsm),
    )
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


@pytest.fixture
def synthetic_pf2_adata():
    return make_synthetic_pf2_data()


@pytest.fixture
def mock_factored_adata():
    return make_mock_factored_adata()
