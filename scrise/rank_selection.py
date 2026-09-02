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
from dataclasses import dataclass

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sps
from parafac2.parafac2 import parafac2_nd
from parafac2.utils import calc_W, condition_slices, project_data
from tqdm import tqdm


def _dense(mat) -> np.ndarray:
    """Return a dense ndarray view of a (possibly sparse) matrix."""
    return mat.toarray() if sps.issparse(mat) else np.asarray(mat)


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


def _bicv_trial(
    X: anndata.AnnData,
    rank: int,
    held_out_cell_frac: float,
    held_out_gene_frac: float,
    rng: np.random.Generator,
    tolerance: float,
    max_iter: int,
) -> float:
    """Run a single bi-cross-validation trial and return the held-out R2X.

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
    """
    cond_idx = X.obs["condition_unique_idxs"].to_numpy().astype(int)
    n_cond = int(cond_idx.max()) + 1
    means = X.var["means"].to_numpy() if "means" in X.var else np.zeros(X.n_vars)

    train_cell_mask = _split_cells_by_condition(cond_idx, held_out_cell_frac, rng)
    test_cell_mask = ~train_cell_mask
    train_gene_mask, test_gene_mask = _split_genes(X.n_vars, held_out_gene_frac, rng)

    X_train = X[train_cell_mask][:, train_gene_mask].copy()
    (weights, (A, B, C), P_train), _ = parafac2_nd(
        X_train,
        rank=rank,
        random_state=int(rng.integers(np.iinfo(np.int32).max)),
        tol=tolerance,
        n_iter_max=max_iter,
    )
    A = A * weights

    # Estimate gene loadings for the held-out genes from the train cells.
    cond_train = cond_idx[train_cell_mask]
    Z = np.concatenate(
        [(P_train[i] @ B) * A[i] for i in range(n_cond) if np.any(cond_train == i)],
        axis=0,
    )
    means_test_genes = means[test_gene_mask]
    X_train_test_genes = (
        _dense(X[train_cell_mask][:, test_gene_mask].X) - means_test_genes
    )
    C_test = np.linalg.lstsq(Z, X_train_test_genes, rcond=None)[0].T

    # Estimate projections for the held-out cells from the train genes.
    means_train_genes = means[train_gene_mask]
    cond_test = cond_idx[test_cell_mask]
    X_test_train_genes = _dense(X[test_cell_mask][:, train_gene_mask].X)
    W_test = calc_W(X_test_train_genes, means_train_genes, C)
    cond_slices_test = condition_slices(cond_test, n_cond)
    P_test, _ = project_data(W_test, [A, B, C], cond_slices_test)

    # Reconstruct and score the held-out test-cell x test-gene block.
    X_test_test_genes = (
        _dense(X[test_cell_mask][:, test_gene_mask].X) - means_test_genes
    )
    ss_res, ss_tot = 0.0, 0.0
    for i in range(n_cond):
        sel = cond_test == i
        if not np.any(sel):
            continue
        actual = X_test_test_genes[sel]
        recon = ((P_test[i] @ B) * A[i]) @ C_test.T
        ss_res += float(np.sum((actual - recon) ** 2))
        ss_tot += float(np.sum(actual**2))

    return 1.0 - ss_res / ss_tot


def bicv(
    X: anndata.AnnData,
    ranks: Sequence[int],
    n_repeats: int = 3,
    held_out_cell_frac: float = 0.2,
    held_out_gene_frac: float = 0.2,
    random_state: int | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 200,
) -> pd.DataFrame:
    """Evaluate rank via bi-cross-validation (BiCV) and in-sample fit R2X.

    For each candidate rank, computes both the ordinary in-sample fit R2X
    (using the full dataset, as in :func:`RISE.factorization.rise_pca_r2x`)
    and the BiCV R2X (repeated ``n_repeats`` times with independent random
    cell/gene splits). The fit R2X increases monotonically with rank; the
    BiCV R2X penalizes overfitting and typically peaks near the rank that
    best generalizes to held-out data. Plot both with
    :func:`RISE.plotting.plot_bicv_r2x` to select a rank.

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

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with columns "Rank", "Repeat", "Metric"
        (one of "Fit R2X" or "BiCV R2X"), and "R2X". Ready to pass to
        :func:`RISE.plotting.plot_bicv_r2x`.
    """
    if not (0 < held_out_cell_frac < 1) or not (0 < held_out_gene_frac < 1):
        raise ValueError(
            "held_out_cell_frac and held_out_gene_frac must both be between 0 and 1."
        )
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

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
        _, fit_r2x = parafac2_nd(
            X,
            rank=rank,
            random_state=int(rng.integers(np.iinfo(np.int32).max)),
            tol=tolerance,
            n_iter_max=max_iter,
        )
        rows.append({"Rank": rank, "Repeat": 0, "Metric": "Fit R2X", "R2X": fit_r2x})

        for repeat in range(n_repeats):
            bicv_r2x = _bicv_trial(
                X,
                rank,
                held_out_cell_frac,
                held_out_gene_frac,
                rng,
                tolerance,
                max_iter,
            )
            rows.append(
                {"Rank": rank, "Repeat": repeat, "Metric": "BiCV R2X", "R2X": bicv_r2x}
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


@dataclass
class BiCVOptimizationResult:
    """Result of :func:`optimize_rank`.

    Attributes
    ----------
    best_rank : int
        The rank with the highest observed mean BiCV R2X.
    best_r2x : float
        The mean BiCV R2X at ``best_rank``.
    history : pandas.DataFrame
        Every rank evaluated during the search, with columns "Rank" (int),
        "R2X" (mean BiCV R2X across repeats), and "R2X_std" (its standard
        deviation across repeats).
    quadratic_fit : numpy.poly1d
        A quadratic (least-squares) fit of BiCV R2X as a function of rank
        over all evaluated points, used to guide the search and available
        for plotting via :func:`RISE.plotting.plot_rank_optimization`.
    quadratic_r2 : float
        The R² of ``quadratic_fit`` against the evaluated points. Low
        values (e.g., well under 0.5) indicate the BiCV R2X curve is not
        well described by a quadratic over the searched range, and
        ``best_rank`` should be trusted over the fit's vertex.
    rank_bounds : tuple of int
        The (low, high) rank bounds searched.
    """

    best_rank: int
    best_r2x: float
    history: pd.DataFrame
    quadratic_fit: np.poly1d
    quadratic_r2: float
    rank_bounds: tuple[int, int]


def _fit_quadratic(
    ranks_seen: np.ndarray, means: np.ndarray
) -> tuple[np.poly1d, float]:
    """Fit BiCV R2X as a quadratic function of rank by ordinary least
    squares, returning the fit and its R² against the observed points."""
    coeffs = np.polyfit(ranks_seen, means, deg=2)
    poly = np.poly1d(coeffs)
    ss_res = float(np.sum((means - poly(ranks_seen)) ** 2))
    ss_tot = float(np.sum((means - means.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return poly, r2


def _quadratic_vertex(poly: np.poly1d) -> float | None:
    """The rank (argmax) of a concave-down quadratic fit, or None if the
    fit is not concave down (i.e., does not describe a single interior
    peak)."""
    a, b, _ = poly.coefficients
    if a >= 0:
        return None
    return -b / (2 * a)


def _propose_next_rank(
    evaluated: dict[int, list[float]], candidates: np.ndarray, lo: int, hi: int
) -> int:
    """Propose the next rank to evaluate, exploiting the expected
    (approximately quadratic, concave) shape of the BiCV R2X curve.

    With at least three ranks evaluated, fits a quadratic to their mean
    BiCV R2X. If that fit is concave down and reasonably explains the
    observations (R² > 0.5) with its vertex inside the search bounds, the
    unevaluated candidate closest to the vertex is proposed -- directly
    homing in on the expected peak. Otherwise (too few points yet, a
    convex/flat fit, or a vertex outside the bounds -- e.g., the curve is
    still monotonic over the searched range) falls back to evaluating the
    midpoint of the widest unsampled gap, to gather the information needed
    for a trustworthy quadratic fit.
    """
    remaining = np.array([r for r in candidates if r not in evaluated])
    if remaining.size == 0:
        return -1

    ranks_seen = np.array(sorted(evaluated))
    if ranks_seen.size >= 3:
        means = np.array([np.mean(evaluated[r]) for r in ranks_seen])
        poly, r2 = _fit_quadratic(ranks_seen, means)
        vertex = _quadratic_vertex(poly)
        if vertex is not None and r2 > 0.5 and lo <= vertex <= hi:
            return int(remaining[np.argmin(np.abs(remaining - vertex))])

    gaps = np.diff(ranks_seen)
    midpoint = (
        (ranks_seen[np.argmax(gaps)] + ranks_seen[np.argmax(gaps) + 1]) / 2
        if gaps.size > 0
        else (lo + hi) / 2
    )
    return int(remaining[np.argmin(np.abs(remaining - midpoint))])


def optimize_rank(
    X: anndata.AnnData,
    rank_bounds: tuple[int, int],
    n_repeats: int = 3,
    held_out_cell_frac: float = 0.2,
    held_out_gene_frac: float = 0.2,
    n_calls: int = 15,
    n_initial_points: int = 5,
    random_state: int | None = None,
    tolerance: float = 1e-6,
    max_iter: int = 200,
) -> BiCVOptimizationResult:
    """Find the rank that maximizes BiCV R2X without evaluating every rank.

    Rather than exhaustively evaluating every rank in ``rank_bounds`` (as
    :func:`bicv` does), this sequentially evaluates a small number of ranks
    (at most ``n_calls``), exploiting the fact that BiCV R2X is expected to
    be an approximately quadratic, concave function of rank: after an
    initial exploration phase, each subsequent rank is chosen to be the
    unevaluated candidate closest to the vertex of a least-squares
    quadratic fit through the ranks evaluated so far. The search stops
    early, before using the full ``n_calls`` budget, once that vertex
    estimate has stabilized for two consecutive rounds. No external
    optimization library or surrogate model (e.g., a Gaussian process) is
    needed for this -- a plain ``numpy.polyfit`` is enough to exploit the
    curve's expected shape.

    Parameters
    ----------
    X : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] and X.var["means"].
    rank_bounds : tuple of int
        (low, high) inclusive bounds on the rank to search. The range must
        span at least 3 ranks.
    n_repeats : int, optional (default: 3)
        Number of independent random cell/gene splits averaged per rank
        evaluation.
    held_out_cell_frac : float, optional (default: 0.2)
        Fraction of cells held out per condition in each BiCV trial.
    held_out_gene_frac : float, optional (default: 0.2)
        Fraction of genes held out in each BiCV trial.
    n_calls : int, optional (default: 15)
        Maximum total number of ranks to evaluate. The search may stop
        earlier once the estimated peak has stabilized.
    n_initial_points : int, optional (default: 5)
        Number of ranks evaluated up front (spread evenly across
        ``rank_bounds``) before switching to the quadratic-guided search.
        Must be >= 3 (the minimum needed for a quadratic fit) and
        <= n_calls.
    random_state : int, optional
        Random seed for reproducibility.
    tolerance : float, optional (default: 1e-6)
        Convergence threshold passed to the PARAFAC2 fit.
    max_iter : int, optional (default: 200)
        Maximum number of iterations passed to the PARAFAC2 fit.

    Returns
    -------
    BiCVOptimizationResult
        The best rank found, its BiCV R2X, the full evaluation history, and
        the quadratic fit used to guide the search (for plotting with
        :func:`RISE.plotting.plot_rank_optimization`).
    """
    lo, hi = int(rank_bounds[0]), int(rank_bounds[1])
    if lo < 1 or hi <= lo:
        raise ValueError(
            "rank_bounds must be an increasing (low, high) pair with low >= 1."
        )
    if hi - lo + 1 < 3:
        raise ValueError("rank_bounds must span at least 3 ranks.")
    if n_initial_points < 3:
        raise ValueError("n_initial_points must be >= 3 (needed for a quadratic fit).")
    if n_calls < n_initial_points:
        raise ValueError("n_calls must be >= n_initial_points.")
    if not (0 < held_out_cell_frac < 1) or not (0 < held_out_gene_frac < 1):
        raise ValueError(
            "held_out_cell_frac and held_out_gene_frac must both be between 0 and 1."
        )

    X = X.to_memory() if hasattr(X, "to_memory") else X
    max_rank = _max_feasible_rank(X, held_out_cell_frac, held_out_gene_frac)
    if hi > max_rank:
        raise ValueError(
            f"rank_bounds upper limit {hi} exceeds the maximum feasible rank "
            f"({max_rank}) given held_out_cell_frac={held_out_cell_frac} and "
            f"held_out_gene_frac={held_out_gene_frac}."
        )

    rng = np.random.default_rng(random_state)
    candidates = np.arange(lo, hi + 1)

    def evaluate(rank: int) -> list[float]:
        return [
            _bicv_trial(
                X,
                rank,
                held_out_cell_frac,
                held_out_gene_frac,
                rng,
                tolerance,
                max_iter,
            )
            for _ in range(n_repeats)
        ]

    evaluated: dict[int, list[float]] = {}
    initial_ranks = np.unique(
        np.round(np.linspace(lo, hi, n_initial_points)).astype(int)
    )
    for rank in tqdm(initial_ranks, desc="BiCV initial exploration"):
        evaluated[int(rank)] = evaluate(int(rank))

    prev_vertex_rank: int | None = None
    stable_rounds = 0
    n_remaining = max(n_calls - len(evaluated), 0)
    for _ in tqdm(range(n_remaining), desc="BiCV quadratic-guided search"):
        next_rank = _propose_next_rank(evaluated, candidates, lo, hi)
        if next_rank < 0:
            break
        evaluated[next_rank] = evaluate(next_rank)

        ranks_seen = np.array(sorted(evaluated))
        means = np.array([np.mean(evaluated[r]) for r in ranks_seen])
        poly, r2 = _fit_quadratic(ranks_seen, means)
        vertex = _quadratic_vertex(poly)
        if vertex is None or r2 <= 0.5:
            stable_rounds = 0
            prev_vertex_rank = None
            continue

        vertex_rank = int(np.clip(round(vertex), lo, hi))
        stable_rounds = stable_rounds + 1 if vertex_rank == prev_vertex_rank else 0
        prev_vertex_rank = vertex_rank
        if stable_rounds >= 2 and len(evaluated) >= n_initial_points + 3:
            break

    ranks_seen = np.array(sorted(evaluated))
    means = np.array([np.mean(evaluated[r]) for r in ranks_seen])
    stds = np.array([np.std(evaluated[r]) for r in ranks_seen])
    poly, r2 = _fit_quadratic(ranks_seen, means)

    best_idx = int(np.argmax(means))
    best_rank = int(ranks_seen[best_idx])
    best_r2x = float(means[best_idx])

    if best_rank in (lo, hi):
        warnings.warn(
            f"optimize_rank: the best rank found ({best_rank}) is at the edge "
            f"of rank_bounds={rank_bounds}. The true optimum may lie outside "
            "the searched range -- consider expanding rank_bounds.",
            stacklevel=2,
        )

    history = (
        pd.DataFrame({"Rank": ranks_seen, "R2X": means, "R2X_std": stds})
        .sort_values("Rank")
        .reset_index(drop=True)
    )

    return BiCVOptimizationResult(
        best_rank=best_rank,
        best_r2x=best_r2x,
        history=history,
        quadratic_fit=poly,
        quadratic_r2=r2,
        rank_bounds=(lo, hi),
    )
