"""
Contract / error-path tests.

Every public function in `scrise.factorization` and `scrise.rank_selection`
documents required AnnData keys and value ranges in its docstring. These
tests pin down what actually happens when a caller violates that contract
(missing keys, wrong dtypes, out-of-range parameters) so a future change
can't silently turn a clear error into a confusing downstream crash (or
worse, a silently wrong result) without a test noticing.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import pytest

from ..factorization import (
    correct_conditions,
    export_factors,
    order_components_by_energy,
    pf2,
)
from ..rank_selection import bicv
from .conftest import make_synthetic_pf2_data


def test_pf2_missing_condition_idxs_raises():
    """pf2() requires X.obs['condition_unique_idxs']; without it the
    caller gets a clear KeyError rather than a cryptic failure deep inside
    the PARAFAC2 solver."""
    X = anndata.AnnData(X=np.random.rand(10, 5).astype(np.float32))
    with pytest.raises(KeyError, match="condition_unique_idxs"):
        pf2(X, rank=2, doEmbedding=False, compress=None)


def test_correct_conditions_missing_condition_idxs_raises():
    X = anndata.AnnData(X=np.random.rand(10, 5).astype(np.float32))
    with pytest.raises(KeyError, match="condition_unique_idxs"):
        correct_conditions(X)


def test_correct_conditions_none_X_raises_typeerror():
    """correct_conditions has an explicit guard for X.X is None (e.g. a
    factors-only AnnData produced by export_factors) -- it should fail
    clearly rather than crash inside `.sum()`."""
    obs = pd.DataFrame({"condition_unique_idxs": [0, 0, 1, 1]})
    X = anndata.AnnData(X=None, obs=obs, shape=(4, 3))
    X.uns["Pf2_A"] = np.ones((2, 2))
    with pytest.raises(TypeError, match="X.X must not be None"):
        correct_conditions(X)


@pytest.mark.parametrize("missing_key", ["Pf2_A", "Pf2_B", "Pf2_weights"])
def test_order_components_by_energy_missing_uns_key_raises(missing_key):
    keys = {"Pf2_A", "Pf2_B", "Pf2_weights"}
    rank, n_genes, n_conditions = 2, 5, 3
    uns = {
        "Pf2_A": np.ones((n_conditions, rank)),
        "Pf2_B": np.ones((rank, rank)),
        "Pf2_weights": np.ones(rank),
    }
    del uns[missing_key]
    X = anndata.AnnData(
        X=np.zeros((6, n_genes), dtype=np.float32),
        uns=uns,
        varm=cast(Mapping[str, Sequence[Any]], {"Pf2_C": np.ones((n_genes, rank))}),
    )
    with pytest.raises(KeyError, match=missing_key):
        order_components_by_energy(X)
    assert missing_key in keys  # sanity: parametrization matches the real keys


@pytest.mark.parametrize(
    "missing",
    ["uns:Pf2_A", "uns:Pf2_B", "uns:Pf2_weights", "varm:Pf2_C", "obsm:projections"],
)
def test_export_factors_missing_any_required_field_raises_keyerror(tmp_path, missing):
    n_cells, n_genes, rank = 10, 6, 2
    fields = {
        "uns:Pf2_A": ("uns", "Pf2_A", np.ones((3, rank))),
        "uns:Pf2_B": ("uns", "Pf2_B", np.ones((rank, rank))),
        "uns:Pf2_weights": ("uns", "Pf2_weights", np.ones(rank)),
        "varm:Pf2_C": ("varm", "Pf2_C", np.ones((n_genes, rank))),
        "obsm:projections": ("obsm", "projections", np.ones((n_cells, rank))),
    }
    uns, varm, obsm = {}, {}, {}
    dest = {"uns": uns, "varm": varm, "obsm": obsm}
    for key, (section, name, value) in fields.items():
        if key == missing:
            continue
        dest[section][name] = value

    X = anndata.AnnData(
        X=np.zeros((n_cells, n_genes), dtype=np.float32), uns=uns, varm=varm, obsm=obsm
    )
    with pytest.raises(KeyError):
        export_factors(X, str(tmp_path / "out.h5ad"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"held_out_cell_frac": 0.0},
        {"held_out_cell_frac": 1.0},
        {"held_out_cell_frac": -0.1},
        {"held_out_gene_frac": 0.0},
        {"held_out_gene_frac": 1.5},
        {"n_repeats": 0},
        {"n_repeats": -1},
    ],
)
def test_bicv_rejects_invalid_arguments(kwargs):
    X = make_synthetic_pf2_data(n_cond=4, n_genes=20, rank=2, seed=0)
    with pytest.raises(ValueError):
        bicv(X, [2], **kwargs)


def test_bicv_rejects_rank_exceeding_max_feasible_rank():
    X = make_synthetic_pf2_data(n_cond=3, n_genes=10, rank=2, seed=0)
    with pytest.raises(ValueError, match="exceeds the maximum feasible rank"):
        bicv(X, [10_000])


def test_export_factors_output_directory_is_created(tmp_path):
    """export_factors should create any missing parent directories for the
    output path rather than failing with FileNotFoundError."""
    n_cells, n_genes, rank = 10, 6, 2
    X = anndata.AnnData(
        X=np.zeros((n_cells, n_genes), dtype=np.float32),
        uns={
            "Pf2_A": np.random.rand(3, rank),
            "Pf2_B": np.random.rand(rank, rank),
            "Pf2_weights": np.random.rand(rank),
        },
        varm=cast(
            Mapping[str, Sequence[Any]], {"Pf2_C": np.random.rand(n_genes, rank)}
        ),
        obsm=cast(
            Mapping[str, Sequence[Any]],
            {"projections": np.random.rand(n_cells, rank).astype(np.float32)},
        ),
    )
    out_path = tmp_path / "nested" / "dir" / "out.h5ad"
    export_factors(X, str(out_path))
    assert out_path.exists()
