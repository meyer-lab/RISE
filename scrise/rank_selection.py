"""
Rank selection for RISE via bi-cross-validation (BiCV).

Bi-cross-validation extends ordinary cross-validation to two-way (row and
column) held-out blocks. For RISE, we hold out a random subset of cells
*and* a random subset of genes, fit PARAFAC2 on the remaining
(train-cell x train-gene) block, and then measure how well the fitted model
predicts the held-out (test-cell x test-gene) block. Unlike the ordinary
in-sample fit R2X (which increases monotonically with rank), the BiCV R2X
penalizes overfitting and typically peaks near the "true" rank of the data.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sps
from parafac2.utils import calc_W, condition_slices, project_data, rmatmul
from tqdm import tqdm

from ._pf2_utils import run_parafac2


def _split_cells_by_condition(
    cond_idx: np.ndarray, held_out_frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Stratified train/test split of cells, holding out a fraction within
    each condition. Every condition retains at least one train cell."""
    train_mask = np.zeros(cond_idx.size, dtype=bool)
    n_cond = int(cond_idx.max()) + 1
    for c in range(n_cond):
        idx = np.flatnonzero(cond_idx == c)
        if idx.size == 0:
            continue
        idx = rng.permutation(idx)
        n_test = min(max(1, round(idx.size * held_out_frac)), idx.size - 1)
        train_mask[idx[n_test:]] = True
    return train_mask


def _split_genes(
    n_genes: int, held_out_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split of gene indices, returning boolean masks."""
    perm = rng.permutation(n_genes)
    n_test = max(1, round(n_genes * held_out_frac))
    test_mask = np.zeros(n_genes, dtype=bool)
    test_mask[perm[:n_test]] = True
    return ~test_mask, test_mask


def _max_feasible_rank(
    X: anndata.AnnData, held_out_cell_frac: float, held_out_gene_frac: float
) -> int:
    """The largest rank for which every condition retains enough train cells
    and enough train genes remain for a BiCV trial to be well-posed."""
    cond_idx = X.obs["condition_unique_idxs"].to_numpy().astype(int)
    n_cond = int(cond_idx.max()) + 1
    min_train_cells = min(
        max(
            1,
            int(np.sum(cond_idx == c))
            - max(1, round(np.sum(cond_idx == c) * held_out_cell_frac)),
        )
        for c in range(n_cond)
    )
    n_train_genes = X.n_vars - max(1, round(X.n_vars * held_out_gene_frac))
    return int(min(min_train_cells, n_train_genes))


def _cell_loadings(
    projections: list[np.ndarray],
    B: np.ndarray,
    A: np.ndarray,
    cond: np.ndarray,
    n_cond: int,
) -> np.ndarray:
    """Per-cell loadings for a set of cells, in those cells' own row order.

    `projections[i]` lists condition ``i``'s cells in their within-condition
    order, so the blocks have to be *scattered back* to the positions those
    cells occupy, not concatenated in condition order. The two agree only when
    conditions happen to be stored as contiguous blocks; on pooled data, where
    conditions are interleaved, concatenating silently misaligns this against
    the expression matrix it is paired with.
    """
    rank = B.shape[1]
    Z = np.empty((cond.size, rank), dtype=np.float64)
    for i in range(n_cond):
        sel = cond == i
        if np.any(sel):
            Z[sel] = (projections[i] @ B) * A[i]
    return Z


#: Nonzeros per row block when streaming column moments. Bounds the per-nonzero
#: temporaries to a few hundred MB regardless of how large the matrix is.
_MOMENT_CHUNK_NNZ = 50_000_000


def _test_block_moments(
    X_mat: Any, cell_mask: np.ndarray, gene_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Column sums and sums of squares over a cell subset, for chosen genes.

    Everything the held-out score needs from the raw data beyond one sparse
    product, computed without materialising the block. Streamed in row blocks
    because the per-nonzero column lookup is otherwise as large as the matrix.
    """
    n_genes = X_mat.shape[1]
    wanted = np.zeros(n_genes, dtype=bool)
    wanted[gene_idx] = True
    sums = np.zeros(n_genes)
    squares = np.zeros(n_genes)

    if not sps.issparse(X_mat):
        block = np.asarray(X_mat)[cell_mask]
        return (
            block.sum(axis=0)[gene_idx],
            np.sum(block.astype(np.float64) ** 2, axis=0)[gene_idx],
        )

    mat = X_mat.tocsr() if X_mat.format != "csr" else X_mat
    n_rows = mat.shape[0]
    start = 0
    while start < n_rows:
        stop = min(
            n_rows,
            max(
                start + 1,
                int(np.searchsorted(mat.indptr, mat.indptr[start] + _MOMENT_CHUNK_NNZ)),
            ),
        )
        lo, hi = int(mat.indptr[start]), int(mat.indptr[stop])
        data = mat.data[lo:hi].astype(np.float64)
        cols = mat.indices[lo:hi]
        keep = wanted[cols] & np.repeat(
            cell_mask[start:stop], np.diff(mat.indptr[start : stop + 1])
        )
        data, cols = data[keep], cols[keep]
        sums += np.bincount(cols, weights=data, minlength=n_genes)
        squares += np.bincount(cols, weights=data**2, minlength=n_genes)
        start = stop

    return sums[gene_idx], squares[gene_idx]


def _bicv_trial(
    X: anndata.AnnData,
    rank: int,
    held_out_cell_frac: float,
    held_out_gene_frac: float,
    seed: int,
    tolerance: float,
    max_iter: int,
    compress: int | tuple[int, int | None] | str | bool | None = "auto",
    compression_kwarg: dict[str, Any] | None = None,
    parafac2_kwarg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run a single bi-cross-validation trial and return its scores and shape.

    Splits cells (stratified by condition) and genes into train/test blocks,
    fits PARAFAC2 on the train-cell x train-gene block, then predicts the
    held-out test-cell x test-gene block:

    - Gene loadings for the held-out genes are estimated by regressing the
      train cells' expression of those genes onto the fitted (train-cell)
      eigen-state projections.
    - Projections for the held-out cells are estimated by fitting the
      PARAFAC2 orthogonal projection for those cells against the fitted
      (train-gene) condition/eigen-state/gene factors.

    R2X is then computed by reconstructing the held-out block from these
    estimates and comparing against the (mean-centered) observed values.

    ``seed`` fully determines the trial: the gene split, the cell split and the
    fit's initialisation all derive from it, so a single trial can be rerun or
    re-scored in isolation from the value reported in the results table.

    Returns
    -------
    dict[str, float]
        ``BiCV R2X`` (the held-out block), ``Train Block R2X`` (in-sample, on
        the block the model was fit to), the four block sizes, and ``Seed``.
    """
    rng = np.random.default_rng(seed)
    cond_idx = X.obs["condition_unique_idxs"].to_numpy().astype(int)
    n_cond = int(cond_idx.max()) + 1
    means = X.var["means"].to_numpy() if "means" in X.var else np.zeros(X.n_vars)

    train_cell_mask = _split_cells_by_condition(cond_idx, held_out_cell_frac, rng)
    test_cell_mask = ~train_cell_mask
    train_gene_mask, test_gene_mask = _split_genes(X.n_vars, held_out_gene_frac, rng)

    X_train = X[train_cell_mask][:, train_gene_mask].copy()
    (weights, (A, B, C), P_train), train_block_r2x = run_parafac2(
        X_train,
        rank=rank,
        random_state=int(rng.integers(np.iinfo(np.int32).max)),
        tol=tolerance,
        n_iter_max=max_iter,
        compress=compress,
        compression_kwarg=compression_kwarg,
        parafac2_kwarg=parafac2_kwarg,
    )
    A = A * weights

    # Everything below reaches the raw data through products against the full
    # matrix, with the cell or gene restriction applied by *zeroing the dense
    # operand*. Slicing instead would copy: on a cohort-scale matrix the three
    # blocks this used to densify are far larger than the matrix itself (a
    # 20% cell / 80% gene block of a 1.2M x 34k dataset is ~50 GB dense), and
    # they were rebuilt for every trial of every rank.
    X_mat = X.X
    n_obs, n_genes = X.shape
    test_gene_idx = np.flatnonzero(test_gene_mask)
    means_test_genes = means[test_gene_mask]

    # Estimate gene loadings for the held-out genes from the train cells.
    #   Z^T (X[train, test] - 1 mu^T) = (Z_full^T X)[:, test] - (Z^T 1) mu^T
    # so one sparse product over the whole matrix replaces the dense block.
    cond_train = cond_idx[train_cell_mask]
    Z = _cell_loadings(P_train, B, A, cond_train, n_cond)
    Z_full = np.zeros((n_obs, Z.shape[1]))
    Z_full[train_cell_mask] = Z
    ZtY = np.asarray(rmatmul(Z_full.T, X_mat), dtype=np.float64)[:, test_gene_idx]
    ZtY -= np.outer(Z.sum(axis=0), means_test_genes)
    # Normal equations rather than `lstsq` on the tall design, which is never
    # formed. `lstsq` on the rank x rank system keeps the minimum-norm
    # behaviour when the fit is rank deficient.
    C_test = np.linalg.lstsq(Z.T @ Z, ZtY, rcond=None)[0].T

    # Estimate projections for the held-out cells from the train genes.
    # A gene factor that is zero on the held-out genes makes `calc_W` ignore
    # them -- including in its `means @ C` centering term -- so the train-gene
    # restriction needs no slice of X.
    C_full = np.zeros((n_genes, C.shape[1]))
    C_full[train_gene_mask] = C
    cond_test = cond_idx[test_cell_mask]
    W_test = calc_W(X_mat, means, C_full)[test_cell_mask]
    cond_slices_test = condition_slices(cond_test, n_cond)
    P_test, _ = project_data(W_test, [A, B, C], cond_slices_test)

    # Score the held-out block. Writing the reconstruction as L @ C_test^T over
    # all test cells lets the three sums be taken from small matrices:
    #   ss_tot   from this block's column moments
    #   cross    from L^T (X[test, test] - 1 mu^T), one more sparse product
    #   ss_fit   from the rank x rank Grams, no data at all
    L = _cell_loadings(P_test, B, A, cond_test, n_cond)
    L_full = np.zeros((n_obs, L.shape[1]))
    L_full[test_cell_mask] = L
    LtY = np.asarray(rmatmul(L_full.T, X_mat), dtype=np.float64)[:, test_gene_idx]
    LtY -= np.outer(L.sum(axis=0), means_test_genes)

    col_sums, col_squares = _test_block_moments(X_mat, test_cell_mask, test_gene_idx)
    n_test_cells = int(test_cell_mask.sum())
    ss_tot = float(
        np.sum(
            col_squares
            - 2.0 * means_test_genes * col_sums
            + n_test_cells * means_test_genes**2
        )
    )
    cross = float(np.sum(C_test.T * LtY))
    ss_fit = float(np.sum((L.T @ L) * (C_test.T @ C_test)))
    ss_res = ss_tot - 2.0 * cross + ss_fit

    return {
        "BiCV R2X": 1.0 - ss_res / ss_tot,
        "Train Block R2X": float(train_block_r2x),
        "NTrainGenes": int(train_gene_mask.sum()),
        "NTestGenes": int(test_gene_mask.sum()),
        "NTrainCells": int(train_cell_mask.sum()),
        "NTestCells": int(test_cell_mask.sum()),
        "Seed": int(seed),
    }


def bicv(
    X: anndata.AnnData | None = None,
    ranks: Sequence[int] | None = None,
    n_repeats: int = 3,
    held_out_cell_frac: float = 0.2,
    held_out_gene_frac: float = 0.2,
    random_state: int | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 200,
    compress: int | tuple[int, int | None] | str | bool | None = "auto",
    compression_kwarg: dict[str, Any] | None = None,
    parafac2_kwarg: dict[str, Any] | None = None,
    condition_key: str | None = None,
    adata: anndata.AnnData | None = None,
) -> pd.DataFrame:
    """Evaluate rank via bi-cross-validation (BiCV) and in-sample fit R2X.

    For each candidate rank, computes both the ordinary in-sample fit R2X
    (using the full dataset, as in :func:`scrise.factorization.rise_pca_r2x`)
    and the BiCV R2X (repeated ``n_repeats`` times with independent random
    cell/gene splits). The fit R2X increases monotonically with rank; the
    BiCV R2X penalizes overfitting and typically peaks near the rank that
    best generalizes to held-out data. Plot both with
    :func:`scrise.plotting.plot_bicv_r2x` to select a rank.

    Parameters
    ----------
    X : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] and X.var["means"]
        (as produced by ``parafac2.normalize.prepare_dataset``).
    ranks : sequence of int
        Candidate rank values to evaluate (e.g., [5, 10, 15, 20, 25, 30]).
    n_repeats : int, optional (default: 3)
        Number of independent random cell/gene splits per rank. Higher
        values give a less noisy BiCV estimate but take longer.
    held_out_cell_frac : float, optional (default: 0.2)
        Fraction of cells held out per condition in each BiCV trial.
    held_out_gene_frac : float, optional (default: 0.2)
        Fraction of genes held out in each BiCV trial.
    random_state : int, optional
        Random seed for reproducibility.
    tolerance : float, optional (default: 1e-6)
        Convergence threshold passed to the PARAFAC2 fit.
    max_iter : int, optional (default: 200)
        Maximum number of iterations passed to the PARAFAC2 fit.
    compress : int | tuple[int, int | None] | str | bool | None, optional
        CANDELINC compression mode passed to each PARAFAC2 fit. Defaults to
        ``"auto"`` (compression dimensions set per rank), which sharply cuts
        the cost of sweeping many ranks and repeats over raw data. Pass
        None/False to fall back to exact ALS.
    compression_kwarg : dict, optional
        Additional keyword arguments forwarded to
        :func:`parafac2.compress.compress_dataset` (e.g. ``n_power_iter``)
        for every trial and in-sample fit. Requires ``compress`` to also be
        set.
    parafac2_kwarg : dict, optional
        Additional keyword arguments forwarded to ``parafac2_nd`` for every
        trial and in-sample fit (e.g. ``normalize_slices``, ``backend``,
        ``n_inner``), for underlying PARAFAC2 options not otherwise exposed
        here. See :func:`scrise.factorization.pf2`'s ``normalize_slices``
        for how it can help with unequal cell counts across conditions.

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with columns "Rank", "Repeat", "Metric" (one of
        "Fit R2X" or "BiCV R2X"), and "R2X". Ready to pass to
        :func:`scrise.plotting.plot_bicv_r2x`.

        BiCV rows carry per-trial diagnostics as additional columns:

        ``Train Block R2X``
            In-sample R2X on the block the model was actually fit to. The
            useful reading of a BiCV curve is that this keeps climbing while
            the held-out R2X turns over; the separate "Fit R2X" metric is a
            different fit on different data and cannot play that role.
        ``NTrainGenes``, ``NTestGenes``, ``NTrainCells``, ``NTestCells``
            The realised block sizes, which set the scale of the spread across
            repeats and confirm the split matches what was requested.
        ``Seed``
            The seed that determines the trial's splits and initialisation, so
            a single trial can be rerun in isolation.

        These columns are NaN on "Fit R2X" rows, which come from an
        unsplit fit on the full dataset.
    """
    if X is None and adata is not None:
        X = adata
    if X is None:
        raise ValueError("Either X or adata must be provided.")
    if ranks is None:
        raise ValueError("ranks must be provided.")

    if not (0 < held_out_cell_frac < 1) or not (0 < held_out_gene_frac < 1):
        raise ValueError(
            "held_out_cell_frac and held_out_gene_frac must both be between 0 and 1."
        )
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    if "condition_unique_idxs" not in X.obs:
        if condition_key is not None and condition_key in X.obs:
            X.obs["condition_unique_idxs"] = pd.Categorical(X.obs[condition_key]).codes
        else:
            raise KeyError(
                "X.obs must contain 'condition_unique_idxs', or provide 'condition_key' pointing to a valid column in X.obs."
            )

    X = X.to_memory() if hasattr(X, "to_memory") else X

    ranks = sorted({int(r) for r in ranks})
    max_rank = _max_feasible_rank(X, held_out_cell_frac, held_out_gene_frac)
    if ranks[-1] > max_rank:
        raise ValueError(
            f"rank {ranks[-1]} exceeds the maximum feasible rank ({max_rank}) given "
            f"held_out_cell_frac={held_out_cell_frac} and "
            f"held_out_gene_frac={held_out_gene_frac}. Test lower ranks, or lower "
            "the held-out fractions."
        )

    rng = np.random.default_rng(random_state)
    rows = []
    for rank in tqdm(ranks, desc="BiCV rank selection"):
        _, fit_r2x = run_parafac2(
            X,
            rank=rank,
            random_state=int(rng.integers(np.iinfo(np.int32).max)),
            tol=tolerance,
            n_iter_max=max_iter,
            compress=compress,
            compression_kwarg=compression_kwarg,
            parafac2_kwarg=parafac2_kwarg,
        )
        # The full-data fit has no split, so the per-trial columns are absent
        # here rather than zero; pandas fills them with NaN.
        rows.append({"Rank": rank, "Repeat": 0, "Metric": "Fit R2X", "R2X": fit_r2x})

        for repeat in range(n_repeats):
            trial_seed = int(rng.integers(np.iinfo(np.int32).max))
            trial = _bicv_trial(
                X,
                rank,
                held_out_cell_frac,
                held_out_gene_frac,
                trial_seed,
                tolerance,
                max_iter,
                compress,
                compression_kwarg,
                parafac2_kwarg,
            )
            rows.append(
                {
                    "Rank": rank,
                    "Repeat": repeat,
                    "Metric": "BiCV R2X",
                    "R2X": trial.pop("BiCV R2X"),
                    **trial,
                }
            )

    results = pd.DataFrame(rows)

    bicv_means = results[results["Metric"] == "BiCV R2X"].groupby("Rank")["R2X"].mean()
    best_rank = int(bicv_means.idxmax())
    if best_rank in (ranks[0], ranks[-1]):
        warnings.warn(
            f"bicv: the rank with the highest mean BiCV R2X ({best_rank}) is at "
            f"the edge of the tested ranks ({ranks[0]}-{ranks[-1]}). The true "
            "optimum may lie outside this range -- consider testing additional "
            "ranks beyond it.",
            stacklevel=2,
        )

    return results
