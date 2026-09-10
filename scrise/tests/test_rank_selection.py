"""
Test bi-cross-validation rank selection.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from ..rank_selection import _bicv_trial, _train_cell_loadings, bicv


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
    assert {"Rank", "Repeat", "Metric", "R2X"} <= set(results.columns)
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


def test_bicv_warns_when_best_rank_is_at_boundary():
    """bicv() should warn if the best tested rank sits at the edge of the
    searched range. With a single rank tested, that rank is trivially both
    ends of the range, so the warning must always fire."""
    X = _make_test_data()

    with pytest.warns(UserWarning, match="edge of the tested ranks"):
        bicv(X, [8], n_repeats=1, random_state=0, max_iter=50)


def test_bicv_condition_key():
    orig = _make_test_data()
    obs = pd.DataFrame(
        {"custom_condition": orig.obs["condition_unique_idxs"].astype(str).to_numpy()}
    )
    X = anndata.AnnData(
        X=orig.X,
        obs=obs,
        var=pd.DataFrame({"means": np.zeros(orig.n_vars)}, index=orig.var_names),
    )

    with pytest.raises(KeyError, match="condition_unique_idxs"):
        bicv(X, [2], n_repeats=1, random_state=0, max_iter=10)

    results = bicv(
        X,
        [2],
        condition_key="custom_condition",
        n_repeats=1,
        random_state=0,
        max_iter=10,
    )
    assert isinstance(results, pd.DataFrame)
    assert "condition_unique_idxs" in X.obs


def test_bicv_parafac2_kwarg_and_compression_kwarg():
    """bicv() should forward parafac2_kwarg/compression_kwarg to every
    in-sample and BiCV-trial PARAFAC2 fit (issue #544)."""
    X = _make_test_data()

    results = bicv(
        X.copy(),
        [2, 4],
        n_repeats=1,
        random_state=0,
        max_iter=20,
        compress="auto",
        parafac2_kwarg={"normalize_slices": True},
        compression_kwarg={"n_power_iter": 1},
    )
    assert isinstance(results, pd.DataFrame)
    assert np.all(np.isfinite(results["R2X"]))

    # compression_kwarg without compress is an error, since there is then
    # no compression step for it to reach.
    with pytest.raises(ValueError, match="compression_kwarg requires compress"):
        bicv(
            X.copy(),
            [2],
            n_repeats=1,
            random_state=0,
            max_iter=20,
            compress=None,
            compression_kwarg={"n_power_iter": 1},
        )


def test_bicv_adata_alias():
    X = _make_test_data()
    results = bicv(adata=X, ranks=[2], n_repeats=1, random_state=0, max_iter=10)
    assert isinstance(results, pd.DataFrame)


# ---------------------------------------------------------------------------
# Alignment of the training-cell loadings (issue #546)
# ---------------------------------------------------------------------------


def _loading_fixture(cond_train: np.ndarray, rank: int = 3):
    """Per-condition projections plus a B and A that make each block identifiable.

    `A[i]` is `(i + 1) * ones`, so a row of the result reveals which condition
    it was built from: any row belonging to condition `i` is exactly `(i + 1)`
    times the corresponding row of `P_train[i] @ B`.
    """
    n_cond = int(cond_train.max()) + 1
    rng = np.random.default_rng(0)
    B = rng.normal(size=(rank, rank))
    A = np.column_stack([np.arange(1, n_cond + 1)] * rank).astype(float)
    P_train = [
        rng.normal(size=(int(np.sum(cond_train == i)), rank)) for i in range(n_cond)
    ]
    return P_train, B, A, n_cond


def test_train_cell_loadings_match_the_rows_they_are_regressed_against():
    """Row k of the loadings must describe the cell at row k of the matrix.

    Regression test for #546. The loadings were built by concatenating
    per-condition blocks in condition order, while the expression they are
    regressed against stays in the cells' natural order. Those two orderings
    coincide only when conditions are stored contiguously, so on pooled data --
    where conditions are interleaved -- the least-squares fit silently paired
    each cell's loading with a different cell's expression.
    """
    cond_train = np.tile(np.arange(4), 25)  # fully interleaved
    P_train, B, A, n_cond = _loading_fixture(cond_train)

    Z = _train_cell_loadings(P_train, B, A, cond_train, n_cond)

    assert Z.shape == (cond_train.size, B.shape[1])
    # Each condition's rows must land where that condition's cells actually are.
    for i in range(n_cond):
        sel = cond_train == i
        np.testing.assert_allclose(Z[sel], (P_train[i] @ B) * A[i])

    # And the identifying scale must survive: a row's magnitude tells you its
    # condition, which is exactly what the misalignment used to scramble.
    for k in range(cond_train.size):
        i = int(cond_train[k])
        np.testing.assert_allclose(
            Z[k], (P_train[i] @ B)[int(np.sum(cond_train[:k] == i))] * A[i]
        )


def test_train_cell_loadings_agree_with_concatenation_when_contiguous():
    """The fix is a no-op on contiguously stored conditions.

    That is the case every previous test used, which is why this went unnoticed:
    concatenating in condition order and scattering by position give the same
    array whenever conditions are already grouped.
    """
    cond_train = np.repeat(np.arange(4), 25)  # contiguous blocks
    P_train, B, A, n_cond = _loading_fixture(cond_train)

    scattered = _train_cell_loadings(P_train, B, A, cond_train, n_cond)
    concatenated = np.concatenate(
        [(P_train[i] @ B) * A[i] for i in range(n_cond)], axis=0
    )
    np.testing.assert_array_equal(scattered, concatenated)


def test_train_cell_loadings_differ_from_concatenation_when_interleaved():
    """...and is *not* a no-op otherwise, which is the whole point.

    Without this, the two tests above would both pass on the unfixed code.
    """
    cond_train = np.tile(np.arange(4), 25)
    P_train, B, A, n_cond = _loading_fixture(cond_train)

    scattered = _train_cell_loadings(P_train, B, A, cond_train, n_cond)
    concatenated = np.concatenate(
        [(P_train[i] @ B) * A[i] for i in range(n_cond)], axis=0
    )
    assert not np.allclose(scattered, concatenated)


def test_train_cell_loadings_handle_a_condition_with_no_training_cells():
    """A condition may be absent from a fold; its rows simply do not exist."""
    cond_train = np.array([0, 2, 0, 2, 2])  # condition 1 has no train cells
    n_cond = 3
    rng = np.random.default_rng(0)
    rank = 2
    B = rng.normal(size=(rank, rank))
    A = rng.normal(size=(n_cond, rank))
    P_train = [
        rng.normal(size=(int(np.sum(cond_train == i)), rank)) for i in range(n_cond)
    ]

    Z = _train_cell_loadings(P_train, B, A, cond_train, n_cond)
    assert Z.shape == (5, rank)
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z[cond_train == 0], (P_train[0] @ B) * A[0])
    np.testing.assert_allclose(Z[cond_train == 2], (P_train[2] @ B) * A[2])


def test_bicv_runs_on_interleaved_conditions():
    """End to end: `bicv` must accept data whose conditions are not grouped."""
    adata = _make_test_data()
    cond = adata.obs["condition_unique_idxs"].to_numpy()
    order = np.argsort(
        [int(np.sum(cond[:k] == cond[k])) for k in range(cond.size)], kind="stable"
    )
    shuffled = adata[order].copy()
    reshuffled = shuffled.obs["condition_unique_idxs"].to_numpy()
    assert int(np.sum(reshuffled[1:] != reshuffled[:-1])) > cond.size // 2

    result = bicv(shuffled, ranks=[2, 3], n_repeats=1, random_state=0, max_iter=50)
    assert np.isfinite(result["R2X"]).all()


# ---------------------------------------------------------------------------
# Per-trial diagnostics (issue #548)
# ---------------------------------------------------------------------------

_TRIAL_COLUMNS = [
    "Train Block R2X",
    "NTrainGenes",
    "NTestGenes",
    "NTrainCells",
    "NTestCells",
    "Seed",
]


def test_bicv_reports_per_trial_diagnostics():
    X = _make_test_data()
    results = bicv(X, [2, 3], n_repeats=2, random_state=0, max_iter=50)

    assert set(_TRIAL_COLUMNS) <= set(results.columns)
    bicv_rows = results[results["Metric"] == "BiCV R2X"]
    assert bicv_rows[_TRIAL_COLUMNS].notna().all().all()


def test_trial_columns_are_absent_on_the_unsplit_fit_rows():
    """ "Fit R2X" is a different fit on the full data, so it has no split."""
    X = _make_test_data()
    results = bicv(X, [2], n_repeats=1, random_state=0, max_iter=50)
    fit_rows = results[results["Metric"] == "Fit R2X"]
    assert fit_rows[_TRIAL_COLUMNS].isna().all().all()


def test_reported_block_sizes_match_the_requested_split():
    X = _make_test_data()
    results = bicv(
        X,
        [2],
        n_repeats=2,
        random_state=0,
        max_iter=50,
        held_out_gene_frac=0.25,
        held_out_cell_frac=0.25,
    )
    rows = results[results["Metric"] == "BiCV R2X"]

    assert (rows["NTrainGenes"] + rows["NTestGenes"] == X.n_vars).all()
    assert (rows["NTrainCells"] + rows["NTestCells"] == X.n_obs).all()
    # Genes split globally, so the count is exact rather than approximate.
    assert (rows["NTestGenes"] == round(X.n_vars * 0.25)).all()
    # Cells split within each condition, so allow for per-condition rounding.
    assert np.allclose(rows["NTestCells"] / X.n_obs, 0.25, atol=0.05)


def test_train_block_r2x_is_in_sample_and_so_climbs_with_rank():
    """The reason this column exists.

    A BiCV curve is only interpretable against the in-sample fit on the *same*
    block: the useful signal is the held-out score turning over while this one
    keeps climbing. The separate "Fit R2X" metric is a different fit on
    different data and cannot play that role.
    """
    X = _make_test_data()
    results = bicv(X, [1, 3, 6], n_repeats=2, random_state=0, max_iter=100)
    rows = results[results["Metric"] == "BiCV R2X"]

    by_rank = rows.groupby("Rank")["Train Block R2X"].mean().sort_index()
    assert np.all(np.diff(by_rank.to_numpy()) >= -1e-6)
    assert (rows["Train Block R2X"] <= 1.0 + 1e-6).all()


def test_reported_seed_reproduces_its_own_trial():
    """A trial can be rerun in isolation from the seed the table reports."""
    X = _make_test_data()
    results = bicv(X, [3], n_repeats=3, random_state=0, max_iter=60)
    rows = results[results["Metric"] == "BiCV R2X"].reset_index(drop=True)

    # Seeds differ across repeats, or "reproducible" would be vacuous.
    assert rows["Seed"].nunique() == len(rows)

    target = rows.iloc[1]
    replayed = _bicv_trial(
        X,
        rank=3,
        held_out_cell_frac=0.2,
        held_out_gene_frac=0.2,
        seed=int(target["Seed"]),
        tolerance=1e-6,
        max_iter=60,
    )
    assert replayed["BiCV R2X"] == pytest.approx(target["R2X"], rel=1e-9)
    assert replayed["Train Block R2X"] == pytest.approx(
        target["Train Block R2X"], rel=1e-9
    )
    assert replayed["NTestGenes"] == target["NTestGenes"]
